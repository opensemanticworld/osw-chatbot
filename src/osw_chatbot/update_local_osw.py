import sys
from importlib.metadata import version
from packaging.specifiers import SpecifierSet
from dotenv import load_dotenv
load_dotenv()  ## fetches environment variables from .env file


from osw.core import OSW
import osw
from osw.auth import CredentialManager
from osw.wtsite import WtSite
from osw.express import osw_download_file, OswExpress


required_schemas = [

                "Category:OSW11a53cdfbdc24524bf8ac435cbf65d9d",  # File   TODO: push this information to github example
                "Category:OSW11a53cdfbdc24524bf8ac435cbf65d9d",  # WikiFile
                "Category:OSW3e3f5dd4f71842fbb8f270e511af8031",  # LocalFile
                "Category:OSW88894b63a51d46b08b5b4b05a6b1b3c3",  # Sample
                "Category:OSW77e749fc598341ac8b6d2fff21574058",  # Software
                "Category:OSW72eae3c8f41f4a22a94dbc01974ed404",  # PrefectFlow
                "Category:OSW92cc6b1a2e6b4bb7bad470dfdcfdaf26",  # Article

                "Category:OSW136953ec4cbf49ef80e2343c0e1981c0",  # PythonEvaluationProcess
            ]


further_schemas = [
]
def update_local_osw(osw_obj):
    print("fetch schemas")
    osw_obj.fetch_schema(
        OSW.FetchSchemaParam(
            schema_title=required_schemas+further_schemas,
            mode="replace",
        )
    )

if __name__ == "__main__":
    osw_obj = OswExpress(domain="mat-o-lab.open-semantic-lab.org")
    update_local_osw(osw_obj)