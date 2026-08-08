from research_lora.data.sampler import SceneBalancedSampler
from research_lora.data.schema import StyleSample


def test_scene_balanced_sampler_is_equal_and_repeatable():
    samples = [StyleSample(f"a{i}", "train", "straight_free_drive", "aggressive") for i in range(2)]
    samples += [StyleSample("b0", "train", "straight_car_follow", "aggressive")]
    sampler = SceneBalancedSampler(samples, "aggressive", seed=3, num_samples=20)
    first = list(sampler); second = list(sampler)
    assert first == second
    assert sampler.epoch_report()["straight_free_drive"] == sampler.epoch_report()["straight_car_follow"] == 0.5


def test_scene_balanced_sampler_can_build_fixed_validation_subset_without_replacement():
    samples = [StyleSample(f"free_{index}", "val", "straight_free_drive", "conservative") for index in range(2)]
    samples += [StyleSample(f"follow_{index}", "val", "straight_car_follow", "conservative") for index in range(5)]
    sampler = SceneBalancedSampler(samples, "conservative", seed=11, replacement=False)
    selected = list(sampler)
    assert len(selected) == 4
    assert len(set(selected)) == 4
    assert sum(index < 2 for index in selected) == 2
    assert sum(index >= 2 for index in selected) == 2
    assert selected == list(sampler)
