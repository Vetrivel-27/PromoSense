from ml_pipeline.src.config.settings import ml_settings


def test_data_paths_configured():
    assert ml_settings.BASE_DIR.exists()
    assert ml_settings.DATA_DIR.exists()
    assert ml_settings.RAW_DATA_DIR.exists()


def test_default_methodological_assumptions():
    assert 0.0 < ml_settings.DEFAULT_ASSUMED_MARGIN_PCT < 1.0
    assert ml_settings.DEFAULT_CAMPAIGN_COST_PER_HH > 0.0
