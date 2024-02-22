import html
import difflib
import random

from rich import print
from rich.table import Table
from rich.console import Console
from rich.rule import Rule
import re

import Levenshtein

# Function to split text into sentences using regex
def split_into_sentences(text):
    sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', text)
    return sentences

class JobPost:

    def __init__(self, id, prompt, ai_generated, edited):
        self.id = id
        self.prompt = prompt
        self.ai_generated = ai_generated
        self.edited = edited

        self.lines1 = split_into_sentences(self.ai_generated)
        self.lines2 = split_into_sentences(self.edited)

        self.diff = self._compute_diff()
        self.stats = self._compute_stats(self.diff)

    def _edit_distance(self):
        return Levenshtein.distance(self.ai_generated, self.edited)

    def _compute_diff(self):
        return list(difflib.ndiff(self.lines1, self.lines2))

    def _compute_stats(self, diff):
        diff = self._compute_diff()
        edit_distance = self._edit_distance()
        additions = sum(1 for line in diff if line.startswith('+'))
        deletions = sum(1 for line in diff if line.startswith('-'))
        unchanged = sum(1 for line in diff if line.startswith(' '))
        return {'line_additions': additions, 
                'line_deletions': deletions, 
                'lines_unchanged': unchanged, 
                'ai_generated_length': len(self.ai_generated),
                'edited_length': len(self.edited),
                'edit_distance': edit_distance}
    
    def _stats_to_html_table(self, stats):
        table_html = '<table style="border: 1px solid black; border-collapse: collapse;">'

        # Add header
        table_html += '''
        <tr>
            <th style="border: 1px solid black; padding: 8px;">Statistic</th>
            <th style="border: 1px solid black; padding: 8px;">Value</th>
        </tr>
        '''

        # Add rows for each statistic
        for key, value in stats.items():
            table_html += f'''
            <tr>
                <td style="border: 1px solid black; padding: 8px;">{html.escape(key)}</td>
                <td style="border: 1px solid black; padding: 8px;">{html.escape(str(value))}</td>
            </tr>
            '''

        # End the table
        table_html += '</table>'

        return table_html


    def table(self, console):
        # Display a rule with a title
        console.rule("[bold red]Example Review", align="center")

        # Create and display the table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Type", justify="right", style="cyan", no_wrap=True)

        table.add_column("Content", style="green")
        table.add_row("ID", self.id)
        table.add_row("", "")  # Separator row
        table.add_row("Prompt", self.prompt)
        table.add_row("", "")  # Separator row
        table.add_row("AI Generated", self.ai_generated)
        table.add_row("", "")  # Separator row
        table.add_row("Edited", self.edited)
        console.print(table)

    # Display a rule before showing changes
        console.rule("[bold blue]Changes", align="center")

    def show_diffs(self, console):
        # Display changes
        for line in self.diff:
            if line.startswith('+'):
                console.print("[green]Added:[/green] " + line[2:])
            elif line.startswith('-'):
                console.print("[red]Removed:[/red] " + line[2:])
            else:
                # Uncomment the next line if you also want to show unchanged lines
                pass
                #console.print("Unchanged: " + line)

    def html_table(self, current_index, max_index):
        # Start the HTML table
        table_html = '<table style="border: 1px solid black; border-collapse: collapse;">'

        # Add header
        table_html += '''
        <tr>
            <th style="border: 1px solid black; padding: 8px;">Type</th>
            <th style="border: 1px solid black; padding: 8px;">Content</th>
        </tr>
        '''

        # Add rows for each field
        rows = [
            ("ID", self.id),
            ("", ""),
            ("Prompt", self.prompt),
            ("", ""),
            ("AI Generated", self.ai_generated),
            ("", ""),
            ("Edited", self.edited)
        ]

        for type, content in rows:
            table_html += f'''
            <tr>
                <td style="border: 1px solid black; padding: 8px; text-align: right;">{html.escape(type)}</td>
                <td style="border: 1px solid black; padding: 8px;">{html.escape(content)}</td>
            </tr>
            '''

        prev_index = max(1, current_index - 1)  # Ensure the index doesn't go below 1
        next_index = min(max_index, current_index + 1)  # Ensure the index doesn't go above max_index
        random_index = random.randint(1, max_index)  # Generate a random index between 1 and max_index

        navigation_html = f'''
        <div style="margin-top: 20px;">
            <a href="/jobpost/{prev_index}"><button>Previous</button></a>
            <a href="/jobpost/{next_index}"><button>Next</button></a>
            <a href="/jobpost/{random_index}"><button>Random</button></a>
        </div>
        '''

        # End the table
        table_html += '</table>'

        #diffs = self.html_diffs()

        stats_html = self._stats_to_html_table(self._compute_stats(self.diff))

        return navigation_html + stats_html + table_html + self.changes_to_html_table(self.get_diff_lines()) 

    def get_diff_lines(self):
        diffs = []

        for line in self.diff:
            if line.startswith('+'):
                diffs.append(("added", line[2:]))
            elif line.startswith('-'):
                diffs.append(("removed", line[2:]))
            else:
                # Uncomment the next line if you also want to include unchanged lines
                pass
                diffs.append(("unchanged", line[2:]))

        return diffs

    def changes_to_html_table(self, changes, include_unchanged=False):
        html_table = '<table border="1">\n'
        html_table += '  <tr>\n'
        html_table += '    <th>Status</th>\n'
        html_table += '    <th>Line</th>\n'
        html_table += '  </tr>\n'

        status_to_style = {
            'added': 'color: green;',
            'removed': 'color: red;',
        }

        for status, line in changes:
            if status == 'unchanged' and not include_unchanged:
                continue
            style = status_to_style.get(status, '')
            html_table += '  <tr>\n'
            html_table += '    <td style="{}">{}</td>\n'.format(style, status)  # Corrected style attribute
            html_table += '    <td>{}</td>\n'.format(line)
            html_table += '  </tr>\n'

        html_table += '</table>'
        return html_table
