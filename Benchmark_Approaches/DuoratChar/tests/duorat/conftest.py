import json
import os
from typing import List, Iterable, Dict

import pytest

import _jsonnet

from duorat.datasets.spider import SpiderDataset
from duorat.utils import registry

# noinspection PyUnresolvedReferences
from duorat import datasets

# noinspection PyUnresolvedReferences
from duorat.preproc import offline, utils

# noinspection PyUnresolvedReferences
from duorat.utils import schema_linker

# noinspection PyUnresolvedReferences
from duorat.asdl.lang import spider


@pytest.fixture(scope="module", params=["configs/duorat/duorat-dev.jsonnet"])
def config(request):
    return json.loads(_jsonnet.evaluate_file(request.param))


@pytest.fixture(scope="module")
def model_preproc(config):
    model_preproc = registry.construct("preproc", config["model"]["preproc"])
    model_preproc.load()
    return model_preproc


@pytest.fixture(scope="module")
def preproc_data(model_preproc):
    return model_preproc.dataset("val")


@pytest.fixture()
def data_prefix() -> str:
    return "data"


@pytest.fixture()
def train_dbs() -> List[str]:
    return [
        "department_management",
        "farm",
        "student_assessment",
        "bike_1",
        "book_2",
        "musical",
        "twitter_1",
        "product_catalog",
        "flight_1",
        "allergy_1",
        "store_1",
        "journal_committee",
        "customers_card_transactions",
        "race_track",
        "coffee_shop",
        "chinook_1",
        "insurance_fnol",
        "medicine_enzyme_interaction",
        "university_basketball",
        "phone_1",
        "match_season",
        "climbing",
        "body_builder",
        "election_representative",
        "apartment_rentals",
        "game_injury",
        "soccer_1",
        "performance_attendance",
        "college_2",
        "debate",
        "insurance_and_eClaims",
        "customers_and_invoices",
        "wedding",
        "theme_gallery",
        "epinions_1",
        "riding_club",
        "gymnast",
        "small_bank_1",
        "browser_web",
        "wrestler",
        "school_finance",
        "protein_institute",
        "cinema",
        "products_for_hire",
        "phone_market",
        "gas_company",
        "party_people",
        "pilot_record",
        "cre_Doc_Control_Systems",
        "company_1",
        "local_govt_in_alabama",
        "formula_1",
        "machine_repair",
        "entrepreneur",
        "perpetrator",
        "csu_1",
        "candidate_poll",
        "movie_1",
        "county_public_safety",
        "inn_1",
        "local_govt_mdm",
        "party_host",
        "storm_record",
        "election",
        "news_report",
        "restaurant_1",
        "customer_deliveries",
        "icfp_1",
        "sakila_1",
        "loan_1",
        "behavior_monitoring",
        "assets_maintenance",
        "station_weather",
        "college_1",
        "sports_competition",
        "manufacturer",
        "hr_1",
        "music_1",
        "baseball_1",
        "mountain_photos",
        "program_share",
        "e_learning",
        "insurance_policies",
        "hospital_1",
        "ship_mission",
        "student_1",
        "company_employee",
        "film_rank",
        "cre_Doc_Tracking_DB",
        "club_1",
        "tracking_grants_for_research",
        "network_2",
        "decoration_competition",
        "document_management",
        "company_office",
        "solvency_ii",
        "entertainment_awards",
        "customers_campaigns_ecommerce",
        "college_3",
        "department_store",
        "aircraft",
        "local_govt_and_lot",
        "school_player",
        "store_product",
        "soccer_2",
        "device",
        "cre_Drama_Workshop_Groups",
        "music_2",
        "manufactory_1",
        "tracking_software_problems",
        "shop_membership",
        "voter_2",
        "products_gen_characteristics",
        "swimming",
        "railway",
        "customers_and_products_contacts",
        "dorm_1",
        "customer_complaints",
        "workshop_paper",
        "tracking_share_transactions",
        "cre_Theme_park",
        "game_1",
        "customers_and_addresses",
        "music_4",
        "roller_coaster",
        "ship_1",
        "city_record",
        "e_government",
        "school_bus",
        "flight_company",
        "cre_Docs_and_Epenses",
        "scientist_1",
        "wine_1",
        "train_station",
        "driving_school",
        "activity_1",
        "flight_4",
        "tracking_orders",
        "architecture",
        "culture_company",
        "geo",
        "scholar",
        "yelp",
        "academic",
        "imdb",
        "restaurants",
    ]


@pytest.fixture()
def val_dbs() -> List[str]:
    return [
        "concert_singer",
        "pets_1",
        "car_1",
        "flight_2",
        "employee_hire_evaluation",
        "cre_Doc_Template_Mgt",
        "course_teach",
        "museum_visit",
        "wta_1",
        "battle_death",
        "student_transcripts_tracking",
        "tvshow",
        "poker_player",
        "voter_1",
        "world_1",
        "orchestra",
        "network_1",
        "dog_kennels",
        "singer",
        "real_estate_properties",
    ]


@pytest.fixture()
def test_dbs() -> List[str]:
    return []


@pytest.fixture()
def duorat_data(
    data_prefix: str,
    train_dbs: Iterable[str],
    val_dbs: Iterable[str],
    test_dbs: Iterable[str],
) -> Dict[str, SpiderDataset]:
    data = {
        "train": registry.construct(
            "dataset",
            {
                "name": "spider",
                "db_path": os.path.join(data_prefix, "database"),
                "paths": [
                    os.path.join(data_prefix, "database", train_db, "examples.json")
                    for train_db in train_dbs
                ],
                "tables_paths": [
                    os.path.join(data_prefix, "database", train_db, "tables.json")
                    for train_db in train_dbs
                ],
            },
        ),
        "val": registry.construct(
            "dataset",
            {
                "name": "spider",
                "db_path": os.path.join(data_prefix, "database"),
                "paths": [
                    os.path.join(data_prefix, "database", val_db, "examples.json")
                    for val_db in val_dbs
                ],
                "tables_paths": [
                    os.path.join(data_prefix, "database", val_db, "tables.json")
                    for val_db in val_dbs
                ],
            },
        ),
        "test": registry.construct(
            "dataset",
            {
                "name": "spider",
                "db_path": os.path.join(data_prefix, "database"),
                "paths": [
                    os.path.join(data_prefix, "database", test_db, "examples.json")
                    for test_db in test_dbs
                ],
                "tables_paths": [
                    os.path.join(data_prefix, "database", test_db, "tables.json")
                    for test_db in test_dbs
                ],
            },
        ),
    }
    for _, dataset in data.items():
        assert isinstance(dataset, SpiderDataset)
    return data
