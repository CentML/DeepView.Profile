

class Position:
    def __init__(self, line, column):
        self.line = line
        self.column = column

    def offset(self, length):
        return Position(self.line, self.column + length)


class SourceMap:
    def __init__(self, source_code):
        self._source_by_line = source_code.splitlines()

    def find_position(self, snippet, line_offset=0):
        for offset, line in enumerate(self._source_by_line[line_offset:]):
            index = line.find(snippet)
            if index == -1:
                continue
            # NOTE: We don't add 1 here to make the line number 0-based
            return Position(line_offset + offset, index)

        return None

    def find_position_on_line(self, snippet, offset_position):
        if offset_position.line >= len(self._source_by_line):
            return None
        index = self._source_by_line[offset_position.line].find(
            snippet, offset_position.column)

        if index == -1:
            return None
        else:
            return Position(offset_position.line, index)
