"""Declare selected approved inputs for the existing GPU1 summary launcher."""
import json
import os
from pathlib import Path
import sys


def input_files(command):
    if len(command) < 3 or Path(command[1]).name not in ('ollama_meeting_summary.py', 'ollama_session_summary.py', 'meeting_chunk_experiment.py'):
        return {}
    transcript_dir = Path(command[2]).expanduser().resolve()
    arguments = command[3:]
    profile, explicit_aliases = 'meeting', None
    extra = {}
    index = 0
    while index < len(arguments):
        arg = arguments[index]
        if arg in ('--profile', '--speaker-aliases', '--chunk-plan', '--chunk-tokenizer', '--tokenizer'):
            if index + 1 >= len(arguments):
                raise ValueError('missing_summary_option_value')
            value = arguments[index + 1]
            if arg == '--profile':
                profile = value
            elif arg == '--speaker-aliases':
                explicit_aliases = value
            else:
                extra['chunk_plan' if arg == '--chunk-plan' else 'chunk_tokenizer'] = {'path': str(Path(value).expanduser().resolve()), 'required': True, 'snapshot': True}
            index += 2
        elif arg.startswith('--profile='):
            profile = arg.split('=', 1)[1]
            index += 1
        elif arg.startswith('--speaker-aliases='):
            explicit_aliases = arg.split('=', 1)[1]
            index += 1
        elif any(arg.startswith(flag + '=') for flag in ('--chunk-plan', '--chunk-tokenizer', '--tokenizer')):
            flag, value = arg.split('=', 1)
            extra['chunk_plan' if flag == '--chunk-plan' else 'chunk_tokenizer'] = {'path': str(Path(value).expanduser().resolve()), 'required': True, 'snapshot': True}
            index += 1
        else:
            index += 1
    if profile != 'meeting':
        return {}
    aliases = (Path(explicit_aliases).expanduser().resolve() if explicit_aliases is not None else
               transcript_dir / 'speaker_aliases.json')
    # An explicitly absent tokenizer must remain absent inside the runner stage.
    if Path(command[1]).name == 'meeting_chunk_experiment.py' or 'chunk_plan' in extra:
        tokenizer = os.environ.get('MEETING_CHUNK_TOKENIZER')
        extra.setdefault('chunk_tokenizer', {'path': str(Path(tokenizer).expanduser().resolve()) if tokenizer else str(transcript_dir / '.no-chunk-tokenizer'), 'required': bool(tokenizer), 'snapshot': True})
    return {**extra, 'approved_aliases': {'path': str(aliases), 'required': explicit_aliases is not None, 'snapshot': True},
            'approved_turn_corrections': {'path': str(transcript_dir / 'speaker_turn_corrections.json'),
                                          'required': False, 'snapshot': True},
            'transcript_index': {'path': str(transcript_dir / 'chunks_out/transcript_chunks.jsonl'),
                                 'required': True, 'snapshot': False}}


if __name__ == '__main__':
    try:
        print(json.dumps(input_files(sys.argv[1:])))
    except (ValueError, OSError):
        print('Cannot declare selected meeting stage inputs', file=sys.stderr)
        raise SystemExit(64)
