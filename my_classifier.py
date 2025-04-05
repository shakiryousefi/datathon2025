import numpy as np
from typing import List, Dict, Any, Tuple
from enum import Enum

label_map = {'Reject': 0, 'Accept': 1}

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


    return all_predictions, all_rejection_reasons, all_explainations

def explain(data: List[Dict]):
    return list(zip(*model(data, explain=True)))

def predict(data: List[Dict]) -> List[int]:
    return model(data, explain=False)[0]

##### HELPER FUNCTIONS #####
def get_nested(data, path):
    keys = path.split('.')
    for key in keys:
        data = data[key]
    return data