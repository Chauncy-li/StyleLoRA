from research_lora.data.schema import StyleSample


def test_existing_straight_scene_index_aliases_are_normalized():
    sample = StyleSample.from_mapping({
        "filename": "cache/sample.npz", "split": "train", "offline_scene_bucket": "straight_car_follow",
        "style": "cons", "sample_quality_valid": True, "split_valid": True,
        "style_performance_vec": [0.1, 0.2, 0.3],
    })
    assert sample.cache_path == "cache/sample.npz"
    assert sample.scene_type == "straight_car_follow" and sample.style == "conservative"
    assert set(sample.metrics) == {"headway_margin", "response_decisiveness", "response_smoothness"}
