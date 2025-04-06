import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from enum import Enum
import re

label_map = {"Reject": 0, "Accept": 1}
inv_label_map = {0: "Reject", 1: "Accept"}


class RejectionReason(Enum):
    PASSPORT_FIRST_NAME_SHOULD_MATCH_ACCOUNT_FORM_FIRST_NAME = 1
    PASSPORT_MIDDLE_NAME_SHOULD_MATCH_ACCOUNT_FORM_MIDDLE_NAME = 2
    PASSPORT_LAST_NAME_SHOULD_MATCH_ACCOUNT_FORM_LAST_NAME = 3
    PASSPORT_NUMBER_SHOULD_MATCH_ACCOUNT_FORM_PASSPORT_NUMBER = 4
    PASSPORT_NUMBER_SHOULD_MATCH_CLIENT_PROFILE_PASSPORT_NUMBER = 5
    PASSPORT_BIRTH_DATE_SHOULD_MATCH_CLIENT_PROFILE_BIRTH_DATE = 6
    PASSPORT_GENDER_SHOULD_MATCH_CLIENT_PROFILE_GENDER = 7
    PASSPORT_NATIONALITY_SHOULD_MATCH_CLIENT_PROFILE_NATIONALITY = 8
    PASSPORT_ISSUE_DATE_SHOULD_MATCH_CLIENT_PROFILE_PASSPORT_ISSUE_DATE = 9
    PASSPORT_EXPIRY_DATE_SHOULD_MATCH_CLIENT_PROFILE_PASSPORT_EXPIRY_DATE = 10
    CLIENT_PROFILE_COUNTRY_OF_DOMICILE_SHOULD_MATCH_ACCOUNT_FORM_COUNTRY_OF_DOMICILE = 11
    CLIENT_PROFILE_EMAIL_ADDRESS_SHOULD_MATCH_ACCOUNT_FORM_EMAIL_ADDRESS = 12
    CLIENT_PROFILE_PHONE_NUMBER_SHOULD_MATCH_ACCOUNT_FORM_PHONE_NUMBER = 13
    CLIENT_PROFILE_CURRENCY_SHOULD_MATCH_ACCOUNT_FORM_CURRENCY = 14
    CLIENT_PROFILE_ADDRESS_SHOULD_MATCH_ACCOUNT_FORM_ADDRESS = 15
    PASSPORT_NUMBER_SHOULD_MATCH_EXTRACTED_MRZ_NUMBER = 16
    EMAIL_ADDRESS_IN_CORRECT_FORMAT = 17
    PHONE_NUMBER_IN_CORRECT_FORMAT = 18
    MARITAL_STATUS_SHOULD_MATCH_FAMILY_BACKGROUND = 19
    PASSPORT_IS_EXPIRED = 20
    CLIENT_PROFILE_SECONDAY_EDUCATION_BETWEEN_SIXTEEN_TWENTYONE = 21
    CLIENT_EARLIEST_EMPLOYMENT_ABOVE_SIXTEEN = 22
    CLIENT_HAS_EMPTYFIELDS = 23
    UNIVERSITY_SHOULD_MATCH_DESCRIPTION = 24
    SECONDARY_EDUCTION_SHOULD_MATCH_EDUCATION_BACKGROUND = 25
    WEALTH_MUST_BE_MENTIONED = 26
    CURRENCY_SHOULD_MATCH_WEALTH_SUMMARY = 27


def model(
    data: List[Dict], explain=False, llm=None, sampling_params=None
) -> Tuple[List[int], List[List[RejectionReason]], List[List[str]]]:
    """
    @return:
     - List[int]: 0 for reject, 1 for accept
     - List[List[RejectionReason]]: rejection reasons (as enums, potentially multiple reasons to reject per sample)
     - List[List[Strings]]: explainations for each rejection
    """
    N = len(data)
    all_predictions = [1] * N  # Default to accept
    all_rejection_reasons = [[] for _ in range(N)]
    all_explainations = [[] for _ in range(N)]

    # Step 1. Fields that should exactly match.
    must_match_fields = [
        {
            "fieldA": "passport.first_name",
            "fieldB": "account_form.first_name",
            "reason": RejectionReason.PASSPORT_FIRST_NAME_SHOULD_MATCH_ACCOUNT_FORM_FIRST_NAME,
        },
        {
            "fieldA": "passport.middle_name",
            "fieldB": "account_form.middle_name",
            "reason": RejectionReason.PASSPORT_MIDDLE_NAME_SHOULD_MATCH_ACCOUNT_FORM_MIDDLE_NAME,
        },
        {
            "fieldA": "passport.last_name",
            "fieldB": "account_form.last_name",
            "reason": RejectionReason.PASSPORT_LAST_NAME_SHOULD_MATCH_ACCOUNT_FORM_LAST_NAME,
        },
        {
            "fieldA": "passport.passport_number",
            "fieldB": "account_form.passport_number",
            "reason": RejectionReason.PASSPORT_NUMBER_SHOULD_MATCH_ACCOUNT_FORM_PASSPORT_NUMBER,
        },
        {
            "fieldA": "passport.passport_number",
            "fieldB": "client_profile.passport_number",
            "reason": RejectionReason.PASSPORT_NUMBER_SHOULD_MATCH_CLIENT_PROFILE_PASSPORT_NUMBER,
        },
        {
            "fieldA": "passport.birth_date",
            "fieldB": "client_profile.birth_date",
            "reason": RejectionReason.PASSPORT_BIRTH_DATE_SHOULD_MATCH_CLIENT_PROFILE_BIRTH_DATE,
        },
        {
            "fieldA": "passport.gender",
            "fieldB": "client_profile.gender",
            "reason": RejectionReason.PASSPORT_GENDER_SHOULD_MATCH_CLIENT_PROFILE_GENDER,
        },
        {
            "fieldA": "passport.nationality",
            "fieldB": "client_profile.nationality",
            "reason": RejectionReason.PASSPORT_NATIONALITY_SHOULD_MATCH_CLIENT_PROFILE_NATIONALITY,
        },
        {
            "fieldA": "passport.passport_issue_date",
            "fieldB": "client_profile.passport_issue_date",
            "reason": RejectionReason.PASSPORT_ISSUE_DATE_SHOULD_MATCH_CLIENT_PROFILE_PASSPORT_ISSUE_DATE,
        },
        {
            "fieldA": "passport.passport_expiry_date",
            "fieldB": "client_profile.passport_expiry_date",
            "reason": RejectionReason.PASSPORT_EXPIRY_DATE_SHOULD_MATCH_CLIENT_PROFILE_PASSPORT_EXPIRY_DATE,
        },
        {
            "fieldA": "client_profile.country_of_domicile",
            "fieldB": "account_form.country_of_domicile",
            "reason": RejectionReason.CLIENT_PROFILE_COUNTRY_OF_DOMICILE_SHOULD_MATCH_ACCOUNT_FORM_COUNTRY_OF_DOMICILE,
        },
        {
            "fieldA": "client_profile.email_address",
            "fieldB": "account_form.email_address",
            "reason": RejectionReason.CLIENT_PROFILE_EMAIL_ADDRESS_SHOULD_MATCH_ACCOUNT_FORM_EMAIL_ADDRESS,
        },
        {
            "fieldA": "client_profile.phone_number",
            "fieldB": "account_form.phone_number",
            "reason": RejectionReason.CLIENT_PROFILE_PHONE_NUMBER_SHOULD_MATCH_ACCOUNT_FORM_PHONE_NUMBER,
        },
        {
            "fieldA": "client_profile.currency",
            "fieldB": "account_form.currency",
            "reason": RejectionReason.CLIENT_PROFILE_CURRENCY_SHOULD_MATCH_ACCOUNT_FORM_CURRENCY,
        },
        {
            "fieldA": "client_profile.address",
            "fieldB": "account_form.address",
            "reason": RejectionReason.CLIENT_PROFILE_ADDRESS_SHOULD_MATCH_ACCOUNT_FORM_ADDRESS,
        },
    ]
    for i, profile in enumerate(data):
        for rule in must_match_fields:
            try:
                if get_nested(profile, rule["fieldA"]) != get_nested(
                    profile, rule["fieldB"]
                ):
                    all_predictions[i] = 0
                    if explain:
                        all_rejection_reasons[i].append(rule["reason"])
                        all_explainations[i].append(
                            f"{rule['fieldA']}({get_nested(profile, rule['fieldA'])}) should match {rule['fieldB']}({get_nested(profile, rule['fieldB'])})"
                        )
            except:
                continue

    # Step 2. Check if passport number matches MRZ
    for i, profile in enumerate(data):
        try:
            passport_number = get_nested(profile, "passport.passport_number")
            mrz_lines = get_nested(profile, "passport.passport_mrz")
            if passport_number and mrz_lines:
                extracted_mrz_number = extract_passport_number_from_mrz(mrz_lines)
                if passport_number != extracted_mrz_number:
                    all_predictions[i] = 0
                    if explain:
                        all_rejection_reasons[i].append(
                            RejectionReason.PASSPORT_NUMBER_SHOULD_MATCH_EXTRACTED_MRZ_NUMBER
                        )
                        all_explainations[i].append(
                            f"Passport number({passport_number}) should match extracted MRZ number({extracted_mrz_number})"
                        )
        except:
            continue

    # Step 3. Check predicate functions
    predicates = {
        RejectionReason.EMAIL_ADDRESS_IN_CORRECT_FORMAT: email_address_in_correct_format,
        RejectionReason.PHONE_NUMBER_IN_CORRECT_FORMAT: phone_number_in_correct_format,
        RejectionReason.MARITAL_STATUS_SHOULD_MATCH_FAMILY_BACKGROUND: marital_status_should_match_family_background,
        RejectionReason.UNIVERSITY_SHOULD_MATCH_DESCRIPTION: universty_should_match_description,
        RejectionReason.PASSPORT_IS_EXPIRED: passport_expired,
        RejectionReason.CLIENT_PROFILE_SECONDAY_EDUCATION_BETWEEN_SIXTEEN_TWENTYONE: secondary_education_interval,
        RejectionReason.CLIENT_HAS_EMPTYFIELDS: contains_invalid_empty_string_wrapper,
        RejectionReason.WEALTH_MUST_BE_MENTIONED: wealth_must_be_mentioned,
        RejectionReason.CURRENCY_SHOULD_MATCH_WEALTH_SUMMARY: currency_should_match_wealth_summary,
        # RejectionReason.CLIENT_EARLIEST_EMPLOYMENT_ABOVE_SIXTEEN: earliest_employment_above_sixteen, <-- This one does not seem to work
    }

    for i, profile in enumerate(data):
        
        for reason, predicate in predicates.items():
            try:
                result, explanation = predicate(profile)
                if not result:
                    all_predictions[i] = 0
                    if explain:
                        all_rejection_reasons[i].append(reason)
                        all_explainations[i].append(explanation)
            except:
                continue

    llm_predicates = {
        RejectionReason.SECONDARY_EDUCTION_SHOULD_MATCH_EDUCATION_BACKGROUND: secondary_education_should_match_education_background,
    }

    if llm:

        for reason, predicate in llm_predicates.items():
            # adjust sampling parameters as necessary for the task

            considered_indices = [
                i for i, profile in enumerate(data) if all_predictions[i] == 1
            ]

            explanation, preds = predicate(
                profiles=[data[i] for i in considered_indices],
                llm=llm,
                sampling_params=sampling_params,
            )

            for idx, res in zip(considered_indices, preds):
                if not res:
                    all_predictions[idx] = 0
                    if explain:
                        all_rejection_reasons[idx].append(reason)
                        all_explainations[idx].append(explanation)

    return all_predictions, all_rejection_reasons, all_explainations


def explain(data: List[Dict]):
    return list(zip(*model(data, explain=True)))


def predict(data: List[Dict]) -> List[int]:
    return model(data, explain=False)[0]


##### REJECTION PREDICATES
# Should all return true, false means we should reject the sample
# May return a string of interest regardless of return value (for debugging)


def email_address_in_correct_format(profile) -> Tuple[bool, str]:
    email = profile["account_form"]["email_address"]
    if not email:
        return False
    email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return (re.match(email_regex, email) is not None), email


def phone_number_in_correct_format(profile) -> Tuple[bool, str]:
    phone = profile["account_form"]["phone_number"]
    normalized = re.sub(r"[ \-()]", "", phone)
    international_pattern = r"^(\+|00)[1-9]\d{7,14}$"  # +41791234567, 0041791234567
    local = r"^\d{8,10}$"
    return (
        bool(
            re.match(international_pattern, normalized) or re.match(local, normalized)
        ),
        phone,
    )


def marital_status_should_match_family_background(profile) -> Tuple[bool, str]:

    family_background_text = profile["client_description"]["Family Background"]
    marriage_status = profile["client_profile"]["marital_status"]

    if (
        "marr" in family_background_text.lower()
        and marriage_status.lower() != "married"
    ):
        return False, "Marital status should be married"
    elif (
        "divorc" in family_background_text.lower()
        and marriage_status.lower() != "divorced"
    ):
        return False, "Marital status should be divorced"
    elif (
        "singl" in family_background_text.lower()
        and marriage_status.lower() != "single"
    ):
        return False, "Marital status should be single"
    elif (
        "widow" in family_background_text.lower()
        and marriage_status.lower() != "widowed"
    ):
        return False, "Marital status should be widowed"

    return True, "Marital status matches family background"


def passport_expired(profile) -> Tuple[bool, str]:
    expiry_str = profile["passport"]["passport_expiry_date"]
    if expiry_str:
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
            if expiry_date < datetime(2025, 4, 1):
                return False, ""
        except ValueError:
            return False, ""

    return True, ""


def secondary_education_interval(profile) -> Tuple[bool, str]:
    sec_age, high_age = get_graduation_ages(profile)
    if sec_age is None or not (16 <= sec_age <= 21):
        return False, ""
    if high_age is not None and not (21 <= high_age <= 26):
        return False, ""

    return True, ""


def contains_invalid_empty_string(profile, parent_key=None) -> bool:
    if isinstance(profile, dict):
        for key, value in profile.items():
            if key == "middle_name" and value == "":
                continue
            if contains_invalid_empty_string(value, key):
                return True
    elif isinstance(profile, list):
        for item in profile:
            if contains_invalid_empty_string(item, parent_key):
                return True
    elif profile == "":
        # Only allow '' if the key is explicitly 'middle_name'
        if parent_key != "middle_name":
            return True
    return False


def contains_invalid_empty_string_wrapper(profile) -> Tuple[bool, str]:
    is_valid = not contains_invalid_empty_string(profile)
    return (
        is_valid,
        (
            "No invalid empty strings (except middle name)"
            if is_valid
            else "Profile contains an empty required field"
        ),
    )


def earliest_employment_above_sixteen(profile) -> Tuple[bool, str]:
    try:
        start_age = get_earliest_employment_start_age(profile)
        if start_age is None or start_age < 6:
            return False, (
                f"Earliest employment start age is {start_age:.1f} (must be at least 12)"
                if start_age is not None
                else "No valid employment start year found"
            )
        return True, f"Earliest employment start age is {start_age:.1f}"
    except Exception as e:
        return False, f"Error checking employment age: {e}"


def universty_should_match_description(profile) -> Tuple[bool, str]:
    university = (
        profile["client_profile"]["higher_education"][0]["university"]
        if profile["client_profile"]["higher_education"]
        else None
    )
    education_background = profile["client_description"]["Education Background"]
    if university is not None:
        if university.lower() not in education_background.lower():
            return False, "University should match description"

    return True, "University matches description"


##### HELPER FUNCTIONS #####
def get_nested(data, path):
    keys = path.split(".")
    for key in keys:
        data = data[key]
    return data


def extract_passport_number_from_mrz(mrz_lines):
    if mrz_lines and len(mrz_lines) >= 2:
        return mrz_lines[1][:9].strip("<")
    return None


def get_graduation_ages(person):
    try:
        birth_date = datetime.strptime(
            person["client_profile"]["birth_date"], "%Y-%m-%d"
        )
    except Exception:
        return None, None  # Invalid birthdate

    try:
        sec_year = person["client_profile"]["secondary_school"]["graduation_year"]
        sec_grad_date = datetime(sec_year, 6, 30)
        sec_age = (sec_grad_date - birth_date).days / 365.25
    except Exception:
        sec_age = None

    higher_ed = person["client_profile"].get("higher_education", [])
    if higher_ed:
        try:
            higher_year = higher_ed[0]["graduation_year"]
            higher_grad_date = datetime(higher_year, 6, 30)
            higher_age = (higher_grad_date - birth_date).days / 365.25
        except Exception:
            higher_age = None
    else:
        higher_age = None

    return sec_age, higher_age


def get_earliest_employment_start_age(person):
    birth_date = datetime.strptime(person["client_profile"]["birth_date"], "%Y-%m-%d")
    employment_history = person["client_profile"].get("employment_history", [])

    start_years = [
        job["start_year"]
        for job in employment_history
        if job.get("start_year") is not None
    ]

    if start_years:
        earliest_start_year = min(start_years)
        employment_start_date = datetime(
            earliest_start_year, 6, 30
        )  # assume mid-year start
        start_age = (employment_start_date - birth_date).days / 365.25
        return start_age
    return None

def secondary_education_should_match_education_background(
    profiles, llm, sampling_params
) -> List[Tuple[bool, str]]:

    prompts = [
        (
            f"Here is the client's secondary school: <{profile['client_profile']['secondary_school']}>\n"
            f"Here is the Education Background of the client: <{profile['client_description']['Education Background']}>\n"
            "Does the secondary school match?\n"
            "If this is the case say YES, otherwise say NO."
        )
        for profile in profiles
    ]

    # Generate texts from the prompts
    outputs = llm.generate(prompts, sampling_params)
    answers = [o.outputs[0].text for o in outputs]

    return [
        not a.startswith(" NO") for a in answers
    ], "Secondary education should match education background"


def wealth_must_be_mentioned(profile) -> Tuple[bool, str]:

    savings = profile["client_profile"]["aum"]["savings"]
    inheritance = profile["client_profile"]["aum"]["inheritance"]

    wealth_summary = profile["client_description"]["Wealth Summary"]

    if savings > 0 and inheritance > 0:
        if str(savings) not in wealth_summary or str(inheritance) not in wealth_summary:
            return False, "Wealth must be mentioned in description"
    elif savings == 0:
        if str(inheritance) not in wealth_summary:
            return False, "Wealth must be mentioned in description"

    if (
        "relationship" in profile["client_profile"]["inheritance_details"]
        and profile["client_profile"]["inheritance_details"]["relationship"]
        not in wealth_summary
    ):
        return False, "Wealth must be mentioned in inheritance description"
    
    if (
        "inheritance year" in profile["client_profile"]["inheritance_details"]
        and str(profile["client_profile"]["inheritance_details"]["inheritance year"])
        not in wealth_summary
    ):
        return False, "Inheritance Year must be mentioned in inheritance description"
    
    if (
        "profession" in profile["client_profile"]["inheritance_details"]
        and str(profile["client_profile"]["inheritance_details"]["inheritance year"])
        not in wealth_summary
    ):
        return False, "Profession must be mentioned in inheritance description"

    return True, "Wealth mentioned in description"

def currency_should_match_wealth_summary(profile) -> Tuple[bool, str]:
    """
    Check if the currency in the wealth summary matches the account form currency.
    """
    account_currency = profile["client_profile"]["currency"]
    wealth_summary = profile["client_description"]["Wealth Summary"]

    if account_currency not in wealth_summary:
        return False, f"Account form currency '{account_currency}' should match wealth summary"

    return True, "Currency matches"
