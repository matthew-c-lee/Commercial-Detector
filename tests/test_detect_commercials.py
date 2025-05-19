from src.detect_commercials import detect_commercials, init_db, get_video_duration_seconds
from pathlib import Path


def test_detect_commercials():
    db = init_db(db_path="tests/cache.db")
    
    output_dir = Path("tests/output")
    # detect_commercials(
    #     conn=db,
    #     video_path=Path("tests/commercials.mkv"),
    #     output_dir=output_dir,
    #     should_reprocess=True,
    #     override_cache=False,
    #     use_cached_llm_response=True
    # )
    detect_commercials(
        conn=db,
        video_path=Path("tests/commercials.mkv"),
        output_dir=output_dir,
        should_reprocess=True,
        override_cache=False,
        use_cached_llm_response=True
    )

    results = []
    for file in output_dir.iterdir():
        results.append((file, get_video_duration_seconds(file)))
        print(file, get_video_duration_seconds(file))

    expected_results = [
        (Path("tests/output/now_thats_what_i_call_music_vol_3.mp4"), 60.093364),
        (Path("tests/output/honeycomb_cereal.mp4"), 30.113415),
        (Path("tests/output/bump_1.mp4"), 5.389),
        (Path("tests/output/bump.mp4"), 5.389),
        (Path("tests/output/trix_cereal.mp4"), 30.130098),
        (Path("tests/output/chuck_e_cheese.mp4"), 15.048366),
        (Path("tests/output/harcourt_learning_direct.mp4"), 60.160097),
        (Path("tests/output/nickelodeon_snow_day_movie.mp4"), 45.378664),
        (Path("tests/output/nickelodeon_promo.mp4"), 60.160097),
    ]

    assert results == expected_results

