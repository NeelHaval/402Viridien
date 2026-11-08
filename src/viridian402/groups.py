"""Shared well identities across interval manifests and the PR54 crop split.

This maps identifiers only. Lithology target names are never canonicalized.
Raw imported manifests and bundle bytes are unchanged.
"""
import re


def normalize_group(group):
    if not isinstance(group,str) or not group:
        raise ValueError('group identity must be nonempty text')
    prefix,separator,value=group.partition(':')
    if not separator:
        return group
    prefix=prefix.lower()
    if prefix=='api':
        return 'api:'+value[:10] if re.fullmatch(r'\d{10,14}',value) else group
    if prefix=='document':
        return 'document:'+re.sub(r'[^a-z0-9]','',value.lower())
    if prefix=='well':
        if re.fullmatch(r'd[-_ ]?582',value,re.I):return 'well:d582'
        mc=re.fullmatch(r'(MC\d+)[_ -]+(P\d+)',value,re.I)
        return 'well:'+mc[1].lower()+'_'+mc[2].lower() if mc else 'well:'+value.lower()
    if prefix=='external':
        return 'external:'+value.lower()
    if prefix!='challenge':
        return group
    if re.search(r'd[-_ ]?582',value,re.I):
        return 'well:d582'
    api=re.match(r'^(\d{14})',value)
    if api:
        return 'api:'+api[1][:10]
    mc=re.match(r'(MC\d+)[_ -]+(P\d+)',value,re.I)
    if mc:
        return 'well:'+mc[1].lower()+'_'+mc[2].lower()
    # Same document-family suffix rule as PR54's published group_key. This
    # handles Callaway, Critter Creek, Rio units, Shell sidetracks, and the
    # atypical twelve-digit Allegheny page without opening source datasets.
    value=re.sub(r'(?i)\.pdf$|[_ ](?:st\d|\d+(?:inch|in_|in\b)|mud|master|log|tvd|md).*','',value)
    return 'document:'+re.sub(r'[^a-z0-9]','',value.lower())
