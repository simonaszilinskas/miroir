"""Model code. It runs only inside Modal containers, never on the Mac.

`miroir.jobs.modal_app` imports this package inside its Modal functions, after the container
has started. It is the one place where torch, transformers, open_clip and
sentence_transformers may be imported (ruff bans them everywhere else), and
`tests/test_no_models_on_the_mac.py` checks that no other module pulls them in.
"""
