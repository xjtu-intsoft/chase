import json
from typing import List, Iterable

from third_party.spider.preprocess.schema import Schema, _get_schemas_from_json
from third_party.spider.process_sql import get_sql


def spider_train_dbs() -> List[str]:
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


def spider_val_dbs() -> List[str]:
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


def _spider_sql_for_spider_database(
    db_id: str, examples: Iterable[dict], tables_data: dict, overwrite_spider_sql: bool
) -> List[dict]:
    res = []
    for i, ex in enumerate(examples):
        try:
            assert overwrite_spider_sql or "sql" not in ex
            query = ex["query"]
            schemas, db_names, tables = _get_schemas_from_json(tables_data)
            schema = schemas[db_id]
            table = tables[db_id]
            schema = Schema(schema, table)
            sql = get_sql(schema, query)
            _ex = dict(ex)
            _ex["sql"] = sql
            res.append(_ex)
        except (KeyError, AssertionError) as e:
            res.append(ex)
        except (RuntimeError, json.decoder.JSONDecodeError) as e:
            raise e
    return res


def spider_sql_for_spider_database(
    db_id: str, overwrite_spider_sql: bool = True
) -> None:
    examples_in_fpath = "data/database/{}/examples.json".format(db_id)
    examples_out_fpath = "data/database/{}/examples.json".format(db_id)
    tables_fpath = "data/database/{}/tables.json".format(db_id)

    with open(examples_in_fpath, "r") as f:
        examples = json.load(f)

    with open(tables_fpath, "r") as f:
        tables_data = json.load(f)

    res = _spider_sql_for_spider_database(
        db_id=db_id,
        examples=examples,
        tables_data=tables_data,
        overwrite_spider_sql=overwrite_spider_sql,
    )

    print(f"Successfully converted {len(res)} queries to Spider SQL")
    with open(examples_out_fpath, "w") as f:
        json.dump(res, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    for db_id in spider_train_dbs():
        spider_sql_for_spider_database(db_id=db_id)
    for db_id in spider_val_dbs():
        spider_sql_for_spider_database(db_id=db_id)
