import re

class FileTemplate:
    ''' Tool class for generating a set of text files from given template.
    
        The file should contain lines with inserting markers defined, such as:
        
            blabla...
            @FT_MARKER.POS1             // define marker POS1

            blabla...
            @FT_MARKER.POS2             // define marker POS2

            // @FT_MARKER.IGNORED       // ignored
            xx @FT_MARKER.IGNORED2      // also ignored
            ...

        In which "POS1", "POS2" are user defined markers.

        Then calling `setMarker(marker, string)` will replace those lines with corresponding texts.
        Unset markers will be kept as is, with original marker defining line commented.

        NOTE: marker defining line should begin with "@FT_MARKER." with optional leading spaces or tabs.
              otherwise it will be ignored by the parser.

    '''

    def __init__(self, template_name, comment_string='//'):
        ''' Init with template file.
        
        '''
        self.m_FileParts = []
        self.m_MarkerDict = {}
        self.m_TemplateName = template_name
        self.m_CommentString = comment_string

        p = re.compile(r'^\s*@FT_MARKER\.(\w+)')

        with open(template_name) as fin:
            # print(f'Loading template file {template_name}...')
            buf = ''
            iline = 0
            for line in fin:
                iline += 1
                res = p.match(line)
                if res is not None:
                    # push buf in and reset buf
                    self.m_FileParts.append(buf)
                    buf = ''

                    marker = res.groups()[0]
                    if marker in self.m_MarkerDict:
                        print(f'  Duplicate marker "{marker}" in line {iline:d}')
                    else:
                        # print(f'  New marker "{marker}" in line {iline:d}')
                        self.m_MarkerDict[marker] = None
                    
                    self.m_FileParts.append((marker, line))
                else:
                    buf += line

            self.m_FileParts.append(buf)

    def setMarker(self, marker, s):
        self.m_MarkerDict[marker] = s
    
    def resetAllMarkers(self):
        for k in self.m_MarkerDict:
            self.m_MarkerDict[k] = None

    def generate(self, outfile, marker_dict=None):
        ''' Generate file with given marker dict.

            If no marker dict given, the internal dict will be used. 
        '''
        if marker_dict is None:
            marker_dict = self.m_MarkerDict

        with open(outfile, 'w') as fout:
            for p in self.m_FileParts:
                if isinstance(p, str):
                    fout.write(p)
                elif isinstance(p, tuple):
                    marker, orig_line = p

                    # original marker line is always written back
                    #   1. mark which source is used in generated file
                    #   2. harmlessly ignored if not set
                    fout.write(self.m_CommentString + ' ' + orig_line)

                    if marker_dict[marker] is not None:
                        fout.write(marker_dict[marker])
                        fout.write('\n') # ensure a newline after the insertion

