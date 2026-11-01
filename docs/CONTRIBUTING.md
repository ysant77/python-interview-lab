# Contributing

All implementations must be original or clearly compatible with repository licensing. Do **not** paste code from proprietary prep platforms or repositories without a license. Problem titles and canonical URLs can be referenced for indexing.

Checklist for each new solution:

1. Include a human-readable invariant/intuition comment and a docstring with `LC <number>` if applicable.
2. Include Big-O time and **auxiliary** space with input dimensions and a note if randomized or amortized.
3. State input assumptions (sorted, unique, nonempty, 0-indexed, mutation, etc.).
4. Add pytest tests for happy path, empties, boundaries, duplicates, and relevant adversarial cases.
5. Change the corresponding catalog `implementation` and `status` **only after code exists**. Distinguish tested and untested solutions.
6. Avoid private interview notes, employer/client identifiers, tokens/credentials and high-volume verbatim copied third-party content.
7. Run `python -m pytest -q` before opening a PR.
