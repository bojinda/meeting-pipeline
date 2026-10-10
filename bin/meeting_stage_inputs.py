"""Declare selected approved inputs for the existing GPU1 summary launcher."""
import json
import os
from pathlib import Path
import sys


def input_files(command):
    if len(command) < 3 or Path(command[1]).name not in ('ollama_meeting_summary.py', 'ollama_session_summary.py', 'meeting_chunk_experiment.py'):
        return {}
    transcript_dir = Path(command[2]).expanduser().resolve()
    arguments = []
    for value in command[3:]:
        arguments.extend(value.split('=', 1) if value.startswith('--') and '=' in value else [value])
    profile, explicit_aliases = 'meeting', None
    checkpoint, checkpoint_sha, editorial_tokenizer = None, None, os.environ.get('MEETING_NOTES_TOKENIZER')
    extra = {}
    index = 0
    while index < len(arguments):
        arg = arguments[index]
        if arg in ('--profile', '--speaker-aliases', '--chunk-plan', '--chunk-tokenizer', '--tokenizer', '--meeting-notes-checkpoint', '--meeting-notes-checkpoint-sha256', '--meeting-notes-tokenizer'):
            if index + 1 >= len(arguments):
                raise ValueError('missing_summary_option_value')
            value = arguments[index + 1]
            if arg == '--profile':
                profile = value
            elif arg == '--speaker-aliases':
                explicit_aliases = value
            elif arg == '--meeting-notes-checkpoint':
                checkpoint = Path(value).expanduser().resolve()
            elif arg == '--meeting-notes-checkpoint-sha256':
                checkpoint_sha = value
            elif arg == '--meeting-notes-tokenizer':
                editorial_tokenizer = value
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
    if checkpoint is not None:
        import hashlib
        if not checkpoint_sha or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_sha:
            raise ValueError('checkpoint_identity_mismatch')
        for role, path in (('editorial_checkpoint', checkpoint), ('editorial_maps', checkpoint.parent / 'chunk_summaries.jsonl'),
                           ('editorial_sections', checkpoint.parent / 'meeting_sections.jsonl')):
            extra[role] = {'path': str(path), 'required': True, 'snapshot': True}
    if '--meeting-notes' in arguments:
        root = Path(__file__).resolve().parents[1]
        for path in (root / 'prompts/meeting').glob('*.txt'):
            extra['editorial_prompt_' + path.stem] = {'path': str(path), 'required': True, 'snapshot': True}
        from meeting_postprocess.editorial_checkpoint import MAP_CODE
        for name in (*MAP_CODE, 'editorial', 'editorial_checkpoint', 'ollama_response'):
            extra['editorial_code_' + name] = {'path': str(root / 'bin/meeting_postprocess' / (name + '.py')), 'required': True, 'snapshot': False}
        extra['editorial_engine'] = {'path': str(root / 'bin/ollama_session_summary.py'), 'required': True, 'snapshot': False}
        if not editorial_tokenizer:
            raise ValueError('editorial_tokenizer_required')
        extra['editorial_tokenizer'] = {'path': str(Path(editorial_tokenizer).expanduser().resolve()), 'required': True, 'snapshot': True}
        settings = os.environ.get('AIHUB_GPU_STAGE_SETTINGS_FILE') or os.environ.get('MEETING_CONFIG_FILE')
        if settings:
            extra['meeting_configuration'] = {'path': str(Path(settings).expanduser().resolve()), 'required': True, 'snapshot': False}
    return {**extra, 'approved_aliases': {'path': str(aliases), 'required': explicit_aliases is not None, 'snapshot': True},
            'approved_turn_corrections': {'path': str(transcript_dir / 'speaker_turn_corrections.json'),
                                          'required': False, 'snapshot': True},
            'transcript_index': {'path': str(transcript_dir / 'chunks_out/transcript_chunks.jsonl'),
                                 'required': True, 'snapshot': checkpoint is not None}}


if __name__ == '__main__':
    try:
        print(json.dumps(input_files(sys.argv[1:])))
    except (ValueError, OSError):
        print('Cannot declare selected meeting stage inputs', file=sys.stderr)
        raise SystemExit(64)
