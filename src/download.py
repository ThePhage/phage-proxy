"""
Downloads and prints votes from the WuFoo form via their API.
"""
from __future__ import division, print_function

import base64
import json
import urllib.request
import urllib.error
import urllib.parse
import os


def download(sub, form, kind='json', password=None):
    req = urllib.request.Request('https://%s.wufoo.com/api/v3/forms/%s/entries.%s' % (sub, form, kind))
    if password is not None:
        req.add_header('Authorization', 'Basic %s' % (base64.encodestring('%s:' % (password)).replace('\n', '')))
    print(req.headers)
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except IOError as e:
        if hasattr(e, 'code'):
            if e.code != 401:
                print('We got another error: %s' % e.code)
        else:
            print(e.headers)
            print(e.headers['www-authenticate'])


votes = download('jebagu',
                 'phage-fractal-vote-experiment',
                 password=os.environ.get('PHAGE_WUFOO_API_KEY'))
print(votes)
