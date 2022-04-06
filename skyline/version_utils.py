import collections
import re

Version = collections.namedtuple('Version', ['major', 'minor', 'patch'])

VERSION_REGEX = re.compile('^\d+\.\d+\.\d+$')


class Version:
    def __init__(self, major, minor, patch):
        self._major = major
        self._minor = minor
        self._patch = patch

    @property
    def major(self):
        return self._major

    @property
    def minor(self):
        return self._minor

    @property
    def patch(self):
        return self._patch

    def __repr__(self):
        return ''.join([
            'Version(major=',
            str(self.major),
            ', minor=',
            str(self.minor),
            ', patch=',
            str(self.patch),
            ')'
        ])

    def __eq__(self, other):
        return (
            self.major == other.major and
            self.minor == other.minor and
            self.patch == other.patch
        )

    def __gt__(self, other):
        self_nums = [self.major, self.minor, self.patch]
        other_nums = [other.major, other.minor, other.patch]

        for self_ver, other_ver in zip(self_nums, other_nums):
            if self_ver > other_ver:
                return True
            elif self_ver < other_ver:
                return False

        return False

    def __ne__(self, other):
        return not (self == other)

    def __ge__(self, other):
        return self == other or self > other

    def __lt__(self, other):
        return not (self >= other)

    def __le__(self, other):
        return not (self > other)

    @classmethod
    def parse_semantic_version(cls, version_str):
        if VERSION_REGEX.match(version_str) is None:
            return None
        version_nums = list(map(int, version_str.split('.')))
        return cls(
            major=version_nums[0],
            minor=version_nums[1],
            patch=version_nums[2],
        )
