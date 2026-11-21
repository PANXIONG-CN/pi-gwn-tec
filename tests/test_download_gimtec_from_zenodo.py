import importlib.util
from pathlib import Path
import unittest


def _load_downloader_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "download_gimtec_from_zenodo.py"
    spec = importlib.util.spec_from_file_location("download_gimtec_from_zenodo", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DownloadGimtecFromZenodoTests(unittest.TestCase):
    def test_metadata_files_are_mapped_to_loader_contract_paths(self):
        mod = _load_downloader_module()
        metadata = {
            "files": [
                {
                    "key": "TEC_2009.npy",
                    "size": 123,
                    "checksum": "md5:abc",
                    "links": {"self": "https://example.test/TEC_2009.npy"},
                },
                {
                    "key": "drivers_2009_2022_2h.npz",
                    "size": 456,
                    "checksum": "md5:def",
                    "links": {"self": "https://example.test/drivers.npz"},
                },
                {
                    "key": "gimtec_metadata.json",
                    "size": 789,
                    "checksum": "md5:ghi",
                    "links": {"self": "https://example.test/gimtec_metadata.json"},
                },
            ]
        }

        planned = mod.plan_downloads(metadata, Path("data/GIMtec"))

        self.assertEqual([p.key for p in planned], ["TEC_2009.npy", "drivers_2009_2022_2h.npz", "gimtec_metadata.json"])
        self.assertEqual(planned[0].destination, Path("data/GIMtec/TEC_2009.npy"))
        self.assertEqual(planned[1].destination, Path("data/GIMtec/drivers/drivers_2009_2022_2h.npz"))
        self.assertEqual(planned[2].destination, Path("data/GIMtec/gimtec_metadata.json"))
        self.assertEqual(planned[1].checksum, "md5:def")

    def test_version_two_drivers_and_extension_years_are_mapped(self):
        mod = _load_downloader_module()
        metadata = {"files": [
            {"key": "TEC_2025.npy", "links": {"self": "https://example.test/TEC_2025.npy"}},
            {"key": "drivers_2009_2025_2h.npz", "links": {"self": "https://example.test/drivers25.npz"}},
        ]}

        planned = mod.plan_downloads(metadata, Path("data/GIMtec"))

        self.assertEqual(planned[0].destination, Path("data/GIMtec/TEC_2025.npy"))
        self.assertEqual(planned[1].destination, Path("data/GIMtec/drivers/drivers_2009_2025_2h.npz"))


if __name__ == "__main__":
    unittest.main()
