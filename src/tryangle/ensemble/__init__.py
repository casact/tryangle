# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
The :mod:`tryangle.ensemble` module includes ensemble-based methods for
VotingChainladder
"""
from ._base import AutoEnsemble # noqa (API Import)
from ._optimizers import SGD, AdaGrad, RMSProp, Adam # noqa (API Import)
