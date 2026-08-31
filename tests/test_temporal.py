from data_sweep.entity_leakage.temporal import detect_aggregation_name_signal


def test_total_prefix_matches():
    assert detect_aggregation_name_signal("total_purchases") is True


def test_avg_prefix_matches():
    assert detect_aggregation_name_signal("avg_response_time") is True


def test_cumulative_prefix_matches():
    assert detect_aggregation_name_signal("cumulative_spend") is True


def test_lifetime_prefix_matches():
    assert detect_aggregation_name_signal("lifetime_value") is True


def test_running_prefix_matches():
    assert detect_aggregation_name_signal("running_total") is True


def test_ytd_prefix_matches():
    assert detect_aggregation_name_signal("ytd_revenue") is True


def test_sum_mean_count_last_max_min_all_match():
    assert detect_aggregation_name_signal("sum_orders") is True
    assert detect_aggregation_name_signal("mean_basket_size") is True
    assert detect_aggregation_name_signal("count_logins") is True
    assert detect_aggregation_name_signal("last_purchase_amount") is True
    assert detect_aggregation_name_signal("max_order_value") is True
    assert detect_aggregation_name_signal("min_order_value") is True


def test_keyword_matches_mid_name_not_just_prefix():
    assert detect_aggregation_name_signal("customer_total_purchases") is True


def test_to_date_suffix_matches():
    assert detect_aggregation_name_signal("purchases_to_date") is True


def test_matches_both_leaked_and_legitimately_windowed_names():
    # naming alone can't discriminate leaked from clean -- both should fire
    assert detect_aggregation_name_signal("total_purchases") is True
    assert detect_aggregation_name_signal("total_purchases_windowed") is True


def test_climax_does_not_false_positive_on_max():
    assert detect_aggregation_name_signal("climax_score") is False


def test_update_date_does_not_false_positive_on_to_date():
    assert detect_aggregation_name_signal("update_date") is False


def test_maximum_does_not_false_positive_on_max():
    assert detect_aggregation_name_signal("maximum_capacity") is False


def test_unrelated_column_name_does_not_match():
    assert detect_aggregation_name_signal("customer_id") is False
    assert detect_aggregation_name_signal("region") is False


def test_custom_keywords_list_respected():
    assert detect_aggregation_name_signal("weird_score", keywords=["weird"]) is True
    assert detect_aggregation_name_signal("total_purchases", keywords=["weird"]) is False
