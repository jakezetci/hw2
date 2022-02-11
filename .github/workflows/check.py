#!/usr/bin/env python3

import json
import sys
from numpy.testing import assert_almost_equal

with open(sys.argv[1]) as json_file:
    data = json.load(json_file)

    assert_almost_equal(data['ron'], 15, 0)
    assert_almost_equal(data['ron_err'], 0.6, 1)
    assert_almost_equal(data['gain'], 2.0, 1)
    assert_almost_equal(data['gain_err'], 0.02, 2)
