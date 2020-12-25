import zipapp
import os

zipapp.create_archive(os.path.join(os.path.dirname(__file__), "mj_bot"))
