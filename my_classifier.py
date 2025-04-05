import numpy as np
from typing import List, Dict, Any, Tuple
from enum import Enum
import re

label_map = {'Reject': 0, 'Accept': 1}
inv_label_map = {0: 'Reject', 1: 'Accept'}

class RejectionReason(Enum):
    PASSPORT_FIRST_NAME_SHOULD_MATCH_ACCOUNT_FORM_FIRST_NAME = 1
    PASSPORT_MIDDLE_NAME_SHOULD_MATCH_ACCOUNT_FORM_MIDDLE_NAME = 2
    PASSPORT_LAST_NAME_SHOULD_MATCH_ACCOUNT_FORM_LAST_NAME = 3
    PASSPORT_NUMBER_SHOULD_MATCH_ACCOUNT_FORM_PASSPORT_NUMBER = 4
    PASSPORT_NUMBER_SHOULD_MATCH_CLIENT_PROFILE_PASSPORT_NUMBER = 5
    PASSPORT_BIRTH_DATE_SHOULD_MATCH_CLIENT_PROFILE_BIRTH_DATE = 7
    PASSPORT_GENDER_SHOULD_MATCH_CLIENT_PROFILE_GENDER = 10
    PASSPORT_NATIONALITY_SHOULD_MATCH_CLIENT_PROFILE_NATIONALITY = 12
    PASSPORT_ISSUE_DATE_SHOULD_MATCH_CLIENT_PROFILE_PASSPORT_ISSUE_DATE = 14
    PASSPORT_EXPIRY_DATE_SHOULD_MATCH_CLIENT_PROFILE_PASSPORT_EXPIRY_DATE = 16
    CLIENT_PROFILE_COUNTRY_OF_DOMICILE_SHOULD_MATCH_ACCOUNT_FORM_COUNTRY_OF_DOMICILE = 18
    CLIENT_PROFILE_EMAIL_ADDRESS_SHOULD_MATCH_ACCOUNT_FORM_EMAIL_ADDRESS = 20
    CLIENT_PROFILE_PHONE_NUMBER_SHOULD_MATCH_ACCOUNT_FORM_PHONE_NUMBER = 22
    CLIENT_PROFILE_CURRENCY_SHOULD_MATCH_ACCOUNT_FORM_CURRENCY = 24
    CLIENT_PROFILE_ADDRESS_SHOULD_MATCH_ACCOUNT_FORM_ADDRESS = 26
    PASSPORT_NUMBER_SHOULD_MATCH_EXTRACTED_MRZ_NUMBER = 27
    EMAIL_ADDRESS_IN_CORRECT_FORMAT = 28
    PHONE_NUMBER_IN_CORRECT_FORMAT = 29
    MARITAL_STATUS_SHOULD_MATCH_FAMILY_BACKGROUND = 30
    UNIVERSITY_SHOULD_MATCH_DESCRIPTION = 31

def model(data: List[Dict], explain=False) -> Tuple[List[int], List[List[RejectionReason]], List[List[str]]]:
    """
    @return:
     - List[int]: 0 for reject, 1 for accept
     - List[List[RejectionReason]]: rejection reasons (as enums, potentially multiple reasons to reject per sample)
     - List[List[Strings]]: explainations for each rejection
    """
    N = len(data)
    all_predictions = [1]*N # Default to accept
    all_rejection_reasons = [[] for _ in range(N)]
    all_explainations = [[] for _ in range(N)]

    # Step 1. Fields that should exactly match.
    must_match_fields = [
        {'fieldA': 'passport.first_name', 'fieldB': 'account_form.first_name', 
        'reason': RejectionReason.PASSPORT_FIRST_NAME_SHOULD_MATCH_ACCOUNT_FORM_FIRST_NAME},
        {'fieldA': 'passport.middle_name', 'fieldB': 'account_form.middle_name', 
        'reason': RejectionReason.PASSPORT_MIDDLE_NAME_SHOULD_MATCH_ACCOUNT_FORM_MIDDLE_NAME},
        {'fieldA': 'passport.last_name', 'fieldB': 'account_form.last_name', 
        'reason': RejectionReason.PASSPORT_LAST_NAME_SHOULD_MATCH_ACCOUNT_FORM_LAST_NAME},
        {'fieldA': 'passport.passport_number', 'fieldB': 'account_form.passport_number', 
        'reason': RejectionReason.PASSPORT_NUMBER_SHOULD_MATCH_ACCOUNT_FORM_PASSPORT_NUMBER},
        {'fieldA': 'passport.passport_number', 'fieldB': 'client_profile.passport_number', 
        'reason': RejectionReason.PASSPORT_NUMBER_SHOULD_MATCH_CLIENT_PROFILE_PASSPORT_NUMBER},
        {'fieldA': 'passport.birth_date', 'fieldB': 'client_profile.birth_date', 
        'reason': RejectionReason.PASSPORT_BIRTH_DATE_SHOULD_MATCH_CLIENT_PROFILE_BIRTH_DATE},
        {'fieldA': 'passport.gender', 'fieldB': 'client_profile.gender', 
        'reason': RejectionReason.PASSPORT_GENDER_SHOULD_MATCH_CLIENT_PROFILE_GENDER},
        {'fieldA': 'passport.nationality', 'fieldB': 'client_profile.nationality', 
        'reason': RejectionReason.PASSPORT_NATIONALITY_SHOULD_MATCH_CLIENT_PROFILE_NATIONALITY},
        {'fieldA': 'passport.passport_issue_date', 'fieldB': 'client_profile.passport_issue_date', 
        'reason': RejectionReason.PASSPORT_ISSUE_DATE_SHOULD_MATCH_CLIENT_PROFILE_PASSPORT_ISSUE_DATE},
        {'fieldA': 'passport.passport_expiry_date', 'fieldB': 'client_profile.passport_expiry_date', 
        'reason': RejectionReason.PASSPORT_EXPIRY_DATE_SHOULD_MATCH_CLIENT_PROFILE_PASSPORT_EXPIRY_DATE},
        {'fieldA': 'client_profile.country_of_domicile', 'fieldB': 'account_form.country_of_domicile', 
        'reason': RejectionReason.CLIENT_PROFILE_COUNTRY_OF_DOMICILE_SHOULD_MATCH_ACCOUNT_FORM_COUNTRY_OF_DOMICILE},
        {'fieldA': 'client_profile.email_address', 'fieldB': 'account_form.email_address', 
        'reason': RejectionReason.CLIENT_PROFILE_EMAIL_ADDRESS_SHOULD_MATCH_ACCOUNT_FORM_EMAIL_ADDRESS},
        {'fieldA': 'client_profile.phone_number', 'fieldB': 'account_form.phone_number', 
        'reason': RejectionReason.CLIENT_PROFILE_PHONE_NUMBER_SHOULD_MATCH_ACCOUNT_FORM_PHONE_NUMBER},
        {'fieldA': 'client_profile.currency', 'fieldB': 'account_form.currency', 
        'reason': RejectionReason.CLIENT_PROFILE_CURRENCY_SHOULD_MATCH_ACCOUNT_FORM_CURRENCY},
        {'fieldA': 'client_profile.address', 'fieldB': 'account_form.address', 
        'reason': RejectionReason.CLIENT_PROFILE_ADDRESS_SHOULD_MATCH_ACCOUNT_FORM_ADDRESS},
        ]
    for i, profile in enumerate(data):
        for rule in must_match_fields:
            if (get_nested(profile, rule['fieldA']) != get_nested(profile, rule['fieldB'])):
                all_predictions[i] = 0
                if explain:
                    all_rejection_reasons[i].append(rule['reason'])
                    all_explainations[i].append(f"{rule['fieldA']}({get_nested(profile, rule['fieldA'])}) should match {rule['fieldB']}({get_nested(profile, rule['fieldB'])})")

    # Step 2. Check if passport number matches MRZ
    for i, profile in enumerate(data):
        passport_number = get_nested(profile, 'passport.passport_number')
        mrz_lines = get_nested(profile, 'passport.passport_mrz')
        if passport_number and mrz_lines:
            extracted_mrz_number = extract_passport_number_from_mrz(mrz_lines)
            if passport_number != extracted_mrz_number:
                all_predictions[i] = 0
                if explain:
                    all_rejection_reasons[i].append(RejectionReason.PASSPORT_NUMBER_SHOULD_MATCH_EXTRACTED_MRZ_NUMBER)
                    all_explainations[i].append(f"Passport number({passport_number}) should match extracted MRZ number({extracted_mrz_number})")

    # Step 3. Check predicate functions
    predicates = {
        RejectionReason.EMAIL_ADDRESS_IN_CORRECT_FORMAT: email_address_in_correct_format,
        RejectionReason.PHONE_NUMBER_IN_CORRECT_FORMAT: phone_number_in_correct_format,
        RejectionReason.MARITAL_STATUS_SHOULD_MATCH_FAMILY_BACKGROUND: marital_status_should_match_family_background,
        RejectionReason.UNIVERSITY_SHOULD_MATCH_DESCRIPTION: universty_should_match_description,
    }
    for i, profile in enumerate(data):
        for reason, predicate in predicates.items():
            result, explanation = predicate(profile)
            if not result:
                all_predictions[i] = 0
                if explain:
                    all_rejection_reasons[i].append(reason)
                    all_explainations[i].append(explanation)

    return all_predictions, all_rejection_reasons, all_explainations

def explain(data: List[Dict]):
    return list(zip(*model(data, explain=True)))

def predict(data: List[Dict]) -> List[int]:
    return model(data, explain=False)[0]

##### REJECTION PREDICATES
# Should all return true, false means we should reject the sample
# May return a string of interest regardless of return value (for debugging)

def email_address_in_correct_format(profile) -> Tuple[bool, str]:
    email = profile['account_form']['email_address']
    if not email:
        return False
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return (re.match(email_regex, email) is not None), email

def phone_number_in_correct_format(profile) -> Tuple[bool, str]:
    phone = profile['account_form']['phone_number']
    normalized = re.sub(r"[ \-()]", "", phone)
    international_pattern = r"^(\+|00)[1-9]\d{7,14}$"  # +41791234567, 0041791234567
    local = r"^\d{8,10}$"
    return bool(re.match(international_pattern, normalized) 
                or re.match(local, normalized)), phone

def marital_status_should_match_family_background(profile) -> Tuple[bool, str]:

    family_background_text = profile['client_description']["Family Background"]
    marriage_status = profile['client_profile']["marital_status"]

    if "marr" in family_background_text.lower() and marriage_status.lower() != "married":
        return False, "Marital status should be married"
    elif "divorc" in family_background_text.lower() and marriage_status.lower() != "divorced":
        return False, "Marital status should be divorced"
    elif "singl" in family_background_text.lower() and marriage_status.lower() != "single":
        return False, "Marital status should be single"
    elif "widow" in family_background_text.lower() and marriage_status.lower() != "widowed":
        return False, "Marital status should be widowed"
    
    return True, "Marital status matches family background"


def universty_should_match_description(profile) -> Tuple[bool, str]:
    university = profile['client_profile']['higher_education'][0]['university'] if profile['client_profile']['higher_education'] else None
    education_background = profile['client_description']["Education Background"]
    if university is not None: 
        if university.lower() not in education_background.lower():
            return False, "University should match description"

    return True, "University matches description"    

##### HELPER FUNCTIONS #####
def get_nested(data, path):
    keys = path.split('.')
    for key in keys:
        data = data[key]
    return data

def extract_passport_number_from_mrz(mrz_lines):
    if mrz_lines and len(mrz_lines) >= 2:
        return mrz_lines[1][:9].strip('<')
    return None
