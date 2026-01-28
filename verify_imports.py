
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_imports")

def check_import(module_name):
    try:
        __import__(module_name)
        logger.info(f"Successfully imported {module_name}")
        return True
    except ImportError as e:
        logger.error(f"Failed to import {module_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error importing {module_name}: {e}")
        return False

modules = [
    "jaxonflow.jax.dispatch",
    "jaxonflow.jax.lowering",
    "jaxonflow.jax.pallas_backend",
    "jaxonflow.pytorch.compiler_backend",
    "jaxonflow.pytorch.custom_ops",
    "jaxonflow.pytorch.triton_wrapper",
    "jaxonflow.pytorch.inductor_extension",
]

success = True
for mod in modules:
    if not check_import(mod):
        success = False

if success:
    logger.info("All modules imported successfully.")
    sys.exit(0)
else:
    logger.error("Some modules failed to import.")
    sys.exit(1)
