from detect_commercials.main import (
    detect_commercials,
    init_db,
    get_video_duration_seconds,
)
from pathlib import Path
import tempfile


def test_detect_commercials() -> None:
    """
    Integration test, but without actually calling out to OpenAI or using PyAnnote
    """
    db = init_db(db_path="tests/test-cache.db")

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = Path(tmpdirname)
        detect_commercials(
            conn=db,
            video_path=Path("tests/commercials.mkv"),
            output_dir=output_dir,
            should_reprocess=True,
            override_cache=False,
            use_cached_llm_response=True,
        )

        results = []
        for file in output_dir.iterdir():
            results.append((file, get_video_duration_seconds(file)))
            print(file, get_video_duration_seconds(file))

        expected_results = [
            (output_dir / "honeycomb_cereal.mp4", 30.113415),
            (output_dir / "bump.mp4", 5.389),
            (output_dir / "trix_cereal.mp4", 30.130098),
            (output_dir / "harcourt_learning_direct.mp4", 60.160097),
            (output_dir / "nickelodeon_promo.mp4", 60.160097),
        ]

        assert sorted(results) == sorted(expected_results)
