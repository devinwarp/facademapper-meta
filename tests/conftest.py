import os

# The suite must not depend on a developer's .env: start with auth off (tests that need it set it themselves).
os.environ["DEMO_PASSWORD"] = ""
