from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from vector.hnsw import HNSW, HNSWNode
from vector.vector_store import VectorStore


class Persistence:
    """
    Handles saving and loading a VectorStore and its optional HNSW index.

    Directory structure:

        index_path/
        ├── manifest.json
        ├── vectors.json
        └── hnsw.json
    """

    VERSION = 1

    MANIFEST_FILE = "manifest.json"
    VECTORS_FILE = "vectors.json"
    HNSW_FILE = "hnsw.json"

    # ================================================================
    # PUBLIC API
    # ================================================================

    @classmethod
    def save(
        cls,
        path: str | Path,
        vector_store: VectorStore,
        hnsw: HNSW | None = None,
    ) -> None:
        """
        Save a VectorStore and an optional HNSW index to disk.

        Existing contents at `path` are replaced.
        """

        path = cls._validate_path(path)

        cls._prepare_directory(path)

        manifest = cls._build_manifest(hnsw)
        vectors_data = cls._serialize_vector_store(vector_store)

        cls._write_json(
            path / cls.MANIFEST_FILE,
            manifest,
        )

        cls._write_json(
            path / cls.VECTORS_FILE,
            vectors_data,
        )

        if hnsw is not None:
            cls._validate_hnsw_matches_store(
                vector_store,
                hnsw,
            )

            hnsw_data = cls._serialize_hnsw(hnsw)

            cls._write_json(
                path / cls.HNSW_FILE,
                hnsw_data,
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> tuple[VectorStore, HNSW | None]:
        """
        Load a VectorStore and optional HNSW index from disk.

        Returns:
            (vector_store, hnsw)

        The returned HNSW value is None when no index was saved.
        """

        path = cls._validate_path(path)

        cls._validate_persistence_directory(path)

        manifest = cls._read_json(
            path / cls.MANIFEST_FILE
        )

        cls._validate_manifest(manifest)

        vectors_data = cls._read_json(
            path / cls.VECTORS_FILE
        )

        vector_store = cls._deserialize_vector_store(
            vectors_data
        )

        has_hnsw = manifest["has_hnsw"]

        if not has_hnsw:
            return vector_store, None

        hnsw_data = cls._read_json(
            path / cls.HNSW_FILE
        )

        hnsw = cls._deserialize_hnsw(
            hnsw_data,
            vector_store,
        )

        return vector_store, hnsw

    # ================================================================
    # MANIFEST
    # ================================================================

    @classmethod
    def _build_manifest(
        cls,
        hnsw: HNSW | None,
    ) -> dict[str, Any]:
        """
        Build metadata describing the persisted index.
        """

        return {
            "version": cls.VERSION,
            "has_hnsw": hnsw is not None,
        }

    @classmethod
    def _validate_manifest(
        cls,
        manifest: dict[str, Any],
    ) -> None:
        """
        Validate the persistence manifest.
        """

        if not isinstance(manifest, dict):
            raise ValueError(
                "invalid persistence manifest"
            )

        if "version" not in manifest:
            raise ValueError(
                "manifest missing version"
            )

        if "has_hnsw" not in manifest:
            raise ValueError(
                "manifest missing has_hnsw"
            )

        version = manifest["version"]

        if version != cls.VERSION:
            raise ValueError(
                f"unsupported persistence version: {version}"
            )

        if not isinstance(manifest["has_hnsw"], bool):
            raise ValueError(
                "manifest has_hnsw must be a boolean"
            )

    # ================================================================
    # VECTOR STORE SERIALIZATION
    # ================================================================

    @classmethod
    def _serialize_vector_store(
        cls,
        vector_store: VectorStore,
    ) -> dict[str, Any]:
        """
        Convert VectorStore state into JSON-serializable data.
        """

        return {
            "metric": vector_store.metric,
            "dimension": vector_store.dimension,
            "vectors": {
                vector_id: list(vector)
                for vector_id, vector
                in vector_store._vectors.items()
            },
            "metadata": vector_store._metadata,
        }

    @classmethod
    def _deserialize_vector_store(
        cls,
        data: dict[str, Any],
    ) -> VectorStore:
        """
        Reconstruct a VectorStore from persisted data.
        """

        cls._validate_vector_store_data(data)

        metric = data["metric"]
        dimension = data["dimension"]
        vectors = data["vectors"]
        metadata = data["metadata"]

        vector_store = VectorStore(
            metric=metric,
            dimension=dimension,
        )

        for vector_id, vector in vectors.items():
            vector_store.add(
                vector_id,
                vector,
                metadata.get(vector_id),
            )

        return vector_store

    # ================================================================
    # HNSW SERIALIZATION
    # ================================================================

    @classmethod
    def _serialize_hnsw(
        cls,
        hnsw: HNSW,
    ) -> dict[str, Any]:
        """
        Convert HNSW graph state into JSON-serializable data.

        Vectors are intentionally NOT stored here because
        VectorStore remains the source of truth.
        """

        nodes: dict[str, Any] = {}

        for node_id, node in hnsw._nodes.items():

            nodes[node_id] = {
                "level": node.level,
                "neighbors": [
                    sorted(neighbors)
                    for neighbors in node.neighbors
                ],
            }

        return {
            "m": hnsw.m,
            "ef_construction": hnsw.ef_construction,
            "ef_search": hnsw.ef_search,
            "metric": hnsw.metric,
            "entry_point": hnsw._entry_point,
            "max_level": hnsw._max_level,
            "dimension": hnsw._dimension,
            "nodes": nodes,
        }

    @classmethod
    def _deserialize_hnsw(
        cls,
        data: dict[str, Any],
        vector_store: VectorStore,
    ) -> HNSW:
        """
        Reconstruct an HNSW index from persisted graph data.
        """

        cls._validate_hnsw_data(
            data,
            vector_store,
        )

        hnsw = HNSW(
            vector_store,
            m=data["m"],
            ef_construction=data["ef_construction"],
            ef_search=data["ef_search"],
            metric=data["metric"],
        )

        nodes_data = data["nodes"]

        for node_id, node_data in nodes_data.items():

            level = node_data["level"]

            node = HNSWNode.create(
                node_id,
                level,
            )

            neighbors_data = node_data["neighbors"]

            for layer, neighbor_ids in enumerate(
                neighbors_data
            ):
                node.neighbors[layer].update(
                    neighbor_ids
                )

            hnsw._nodes[node_id] = node

        hnsw._entry_point = data["entry_point"]
        hnsw._max_level = data["max_level"]
        hnsw._dimension = data["dimension"]

        return hnsw

    # ================================================================
    # VECTOR STORE VALIDATION
    # ================================================================

    @classmethod
    def _validate_vector_store_data(
        cls,
        data: dict[str, Any],
    ) -> None:
        """
        Validate persisted VectorStore data before reconstruction.
        """

        if not isinstance(data, dict):
            raise ValueError(
                "invalid VectorStore persistence data"
            )

        required = {
            "metric",
            "dimension",
            "vectors",
            "metadata",
        }

        missing = required - data.keys()

        if missing:
            raise ValueError(
                f"VectorStore data missing fields: {missing}"
            )

        vectors = data["vectors"]
        metadata = data["metadata"]

        if not isinstance(vectors, dict):
            raise ValueError(
                "vectors must be a dictionary"
            )

        if not isinstance(metadata, dict):
            raise ValueError(
                "metadata must be a dictionary"
            )

        dimension = data["dimension"]

        if dimension is not None:
            if (
                not isinstance(dimension, int)
                or dimension <= 0
            ):
                raise ValueError(
                    "invalid VectorStore dimension"
                )

        for vector_id, vector in vectors.items():

            if not isinstance(vector_id, str):
                raise ValueError(
                    "vector IDs must be strings"
                )

            if not isinstance(vector, list):
                raise ValueError(
                    f"vector for {vector_id!r} "
                    "must be a list"
                )

            if (
                dimension is not None
                and len(vector) != dimension
            ):
                raise ValueError(
                    f"vector {vector_id!r} has "
                    "incorrect dimension"
                )

        for vector_id in metadata:

            if vector_id not in vectors:
                raise ValueError(
                    f"metadata exists for unknown "
                    f"vector {vector_id!r}"
                )

    # ================================================================
    # HNSW VALIDATION
    # ================================================================

    @classmethod
    def _validate_hnsw_matches_store(
        cls,
        vector_store: VectorStore,
        hnsw: HNSW,
    ) -> None:
        """
        Validate that the HNSW index belongs to the supplied VectorStore.
        """

        if hnsw.metric != vector_store.metric:
            raise ValueError(
                "HNSW metric does not match VectorStore metric"
            )

        if (
            hnsw._dimension is not None
            and vector_store.dimension is not None
            and hnsw._dimension != vector_store.dimension
        ):
            raise ValueError(
                "HNSW dimension does not match "
                "VectorStore dimension"
            )

        for node_id in hnsw._nodes:

            if not vector_store.contains(node_id):
                raise ValueError(
                    f"HNSW node {node_id!r} "
                    "does not exist in VectorStore"
                )

    @classmethod
    def _validate_hnsw_data(
        cls,
        data: dict[str, Any],
        vector_store: VectorStore,
    ) -> None:
        """
        Validate persisted HNSW graph data before reconstruction.
        """

        if not isinstance(data, dict):
            raise ValueError(
                "invalid HNSW persistence data"
            )

        required = {
            "m",
            "ef_construction",
            "ef_search",
            "metric",
            "entry_point",
            "max_level",
            "dimension",
            "nodes",
        }

        missing = required - data.keys()

        if missing:
            raise ValueError(
                f"HNSW data missing fields: {missing}"
            )

        nodes = data["nodes"]

        if not isinstance(nodes, dict):
            raise ValueError(
                "HNSW nodes must be a dictionary"
            )

        if data["metric"] != vector_store.metric:
            raise ValueError(
                "persisted HNSW metric does not match "
                "VectorStore metric"
            )

        dimension = data["dimension"]

        if (
            dimension is not None
            and vector_store.dimension is not None
            and dimension != vector_store.dimension
        ):
            raise ValueError(
                "persisted HNSW dimension does not match "
                "VectorStore dimension"
            )

        for node_id, node_data in nodes.items():

            if node_id not in vector_store:
                raise ValueError(
                    f"HNSW node {node_id!r} "
                    "does not exist in VectorStore"
                )

            cls._validate_hnsw_node(
                node_id,
                node_data,
                nodes,
            )

        entry_point = data["entry_point"]

        if nodes:

            if entry_point is None:
                raise ValueError(
                    "HNSW has nodes but no entry point"
                )

            if entry_point not in nodes:
                raise ValueError(
                    "HNSW entry point does not exist"
                )

            actual_max_level = max(
                node_data["level"]
                for node_data in nodes.values()
            )

            if data["max_level"] != actual_max_level:
                raise ValueError(
                    "HNSW max_level does not match nodes"
                )

            if (
                nodes[entry_point]["level"]
                != actual_max_level
            ):
                raise ValueError(
                    "HNSW entry point is not "
                    "at the maximum level"
                )

        else:

            if entry_point is not None:
                raise ValueError(
                    "empty HNSW must not have an entry point"
                )

            if data["max_level"] != -1:
                raise ValueError(
                    "empty HNSW must have max_level = -1"
                )

    @classmethod
    def _validate_hnsw_node(
        cls,
        node_id: str,
        node_data: Any,
        all_nodes: dict[str, Any],
    ) -> None:
        """
        Validate one persisted HNSW node.
        """

        if not isinstance(node_data, dict):
            raise ValueError(
                f"invalid HNSW node data for {node_id!r}"
            )

        if "level" not in node_data:
            raise ValueError(
                f"HNSW node {node_id!r} missing level"
            )

        if "neighbors" not in node_data:
            raise ValueError(
                f"HNSW node {node_id!r} "
                "missing neighbors"
            )

        level = node_data["level"]
        neighbors = node_data["neighbors"]

        if (
            not isinstance(level, int)
            or level < 0
        ):
            raise ValueError(
                f"invalid level for node {node_id!r}"
            )

        if not isinstance(neighbors, list):
            raise ValueError(
                f"neighbors for node {node_id!r} "
                "must be a list"
            )

        if len(neighbors) != level + 1:
            raise ValueError(
                f"node {node_id!r} has incorrect "
                "number of layers"
            )

        for layer, neighbor_ids in enumerate(
            neighbors
        ):

            if not isinstance(neighbor_ids, list):
                raise ValueError(
                    f"neighbors for node {node_id!r} "
                    f"at layer {layer} must be a list"
                )


            for neighbor_id in neighbor_ids:

                if neighbor_id not in all_nodes:
                    raise ValueError(
                        f"node {node_id!r} references "
                        f"unknown neighbor {neighbor_id!r}"
                    )

                neighbor_data = all_nodes[neighbor_id]

                if neighbor_data["level"] < layer:
                    raise ValueError(
                        f"neighbor {neighbor_id!r} "
                        f"does not exist at layer {layer}"
                    )

    # ================================================================
    # FILE / DIRECTORY HELPERS
    # ================================================================

    @staticmethod
    def _validate_path(
        path: str | Path,
    ) -> Path:
        """
        Validate and normalize a persistence path.
        """

        if not isinstance(path, (str, Path)):
            raise TypeError(
                "path must be a string or pathlib.Path"
            )

        path = Path(path)

        if not str(path):
            raise ValueError(
                "path must not be empty"
            )

        return path

    @classmethod
    def _prepare_directory(
        cls,
        path: Path,
    ) -> None:
        """
        Create a clean persistence directory.
        """

        if path.exists():

            if path.is_file():
                raise ValueError(
                    f"persistence path {path} "
                    "is an existing file"
                )

            shutil.rmtree(path)

        path.mkdir(
            parents=True,
            exist_ok=False,
        )

    @classmethod
    def _validate_persistence_directory(
        cls,
        path: Path,
    ) -> None:
        """
        Ensure a valid persistence directory exists.
        """

        if not path.exists():
            raise FileNotFoundError(
                f"persistence path does not exist: {path}"
            )

        if not path.is_dir():
            raise ValueError(
                f"persistence path is not a directory: {path}"
            )

        manifest_path = path / cls.MANIFEST_FILE
        vectors_path = path / cls.VECTORS_FILE

        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"missing {cls.MANIFEST_FILE}"
            )

        if not vectors_path.is_file():
            raise FileNotFoundError(
                f"missing {cls.VECTORS_FILE}"
            )

    @staticmethod
    def _write_json(
        path: Path,
        data: dict[str, Any],
    ) -> None:
        """
        Write JSON data to disk.
        """

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def _read_json(
        path: Path,
    ) -> dict[str, Any]:
        """
        Read JSON data from disk.
        """

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

        except json.JSONDecodeError as error:

            raise ValueError(
                f"invalid JSON in {path}"
            ) from error

        if not isinstance(data, dict):
            raise ValueError(
                f"expected JSON object in {path}"
            )

        return data