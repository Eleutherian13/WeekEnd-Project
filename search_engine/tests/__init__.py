"""
tests
=====

Test suite root. Structure MIRRORS the top-level package structure
exactly (tests/analysis for search_engine/analysis, etc.) so that any
engineer can find a module's tests without guessing.

Testing conventions for this project (enforced from Milestone 1 onward):
  - Every public class/function gets unit tests covering: the happy path,
    at least one edge case, and at least one invalid-input case.
  - Data structure invariants (e.g. "PostingsList stays sorted") get an
    explicit test, not just incidental coverage via a higher-level test.
  - Framework: pytest.
"""
