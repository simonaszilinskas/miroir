"""Model code. It runs only on a Scaleway GPU machine, never on the Mac.

The worker (`miroir.remote.worker`, started by the hidden `miroir scw serve`) and the demo
site (`miroir.remote.site`) import it there. It is the one place where torch, transformers,
open_clip and sentence_transformers may be imported (ruff bans them everywhere else), and
`tests/test_no_models_on_the_mac.py` checks that no other module pulls them in.
"""
