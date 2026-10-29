import hashlib
import json
import base64
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools.audit_public import audit, binary_errors, copied_blocks, guide_overlaps, media_metadata_bytes, media_tag_errors, text_errors


def manifest(root):
    target = root / 'artifacts/SHA256SUMS'
    target.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob('*') if p.is_file() and not p.is_symlink()
                   and p != target and '.git' not in p.relative_to(root).parts)
    target.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n' for p in files))


class PublicAuditTests(unittest.TestCase):
    def test_sensitive_content_and_binary_encoding(self):
        self.assertTrue(text_errors('/' + 'home/user/work'))
        self.assertTrue(text_errors('TASK-' + '363'))
        self.assertTrue(text_errors('person' + '@example.com'))
        self.assertFalse(text_errors('python evaluate.py --ctrl my_servo --detector my_detector'))
        secret = '/' + 'home/test/path'
        self.assertIn('machine home path', binary_errors(secret.encode()))
        self.assertIn('machine home path', binary_errors(secret.encode('utf-16-le')))
        self.assertIn('machine home path', binary_errors(secret.encode('utf-16-be')))

    def test_manifest_coverage_and_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'README.md').write_text('Our code only\n'); manifest(root)
            self.assertEqual(audit(root)['status'], 'pass')
            (root/'new.py').write_text('extra')
            self.assertIn('checksum content/coverage mismatch', audit(root)['errors'])
            (root/'new.py').unlink(); (root/'README.md').write_text('changed')
            self.assertIn('checksum content/coverage mismatch', audit(root)['errors'])
            (root/'evaluate.py').write_text('kit file'); manifest(root)
            self.assertTrue(any('excluded' in e for e in audit(root)['errors']))

    def test_manifest_rejects_duplicate_unsafe_and_noncanonical_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'README.md').write_text('ok'); manifest(root)
            path=root/'artifacts/SHA256SUMS'; first=path.read_text()
            for extra in (first, '0'*64+'  ../README.md\n', '0'*64+'  /tmp/name\n',
                          '0'*64+'  a/./file\n', '0'*64+'  a\\file\n',
                          '0'*64+'  artifacts/SHA256SUMS\n'):
                path.write_text(first+extra)
                self.assertIn('malformed checksum manifest', audit(root)['errors'])

    def test_symlink_file_directory_and_git_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'README.md').write_text('ok'); manifest(root)
            for name,target in [('file-link','README.md'), ('dir-link','artifacts'), ('.git','artifacts')]:
                link=root/name;link.symlink_to(target,target_is_directory=name!='file-link')
                self.assertTrue(any('symlink:' in e for e in audit(root)['errors']), name)
                link.unlink()

    def test_six_line_source_block_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            outer=Path(directory);kit=outer/'kit';root=outer/'public';kit.mkdir();root.mkdir()
            lines='\n'.join('value_'+str(i)+' = '+str(i*i) for i in range(6))+'\n'
            (kit/'run.py').write_text(lines); (root/'ours.py').write_text(lines)
            self.assertEqual(len(copied_blocks(root,kit)),1)
            self.assertEqual(audit(root,kit=kit,require_checksums=False)['status'],'FAIL')

    def test_reflowed_guide_prose_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            outer=Path(directory);root=outer/'public';root.mkdir();guide=outer/'guide.txt'
            words=['문장단어'+str(i)+'가나다라마바사' for i in range(14)]
            guide.write_text(' '.join(words)); (root/'README.md').write_text('\n'.join(words))
            self.assertEqual(len(guide_overlaps(root,guide)),1)
            self.assertEqual(audit(root,guide=guide,require_checksums=False)['status'],'FAIL')

    def test_missing_external_comparison_sources_are_not_reported_as_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=audit(root,kit=root/'absent',guide=root/'missing',require_checksums=False)
            self.assertEqual(result['status'],'FAIL')
            self.assertFalse(result['kit_comparison_performed'])
            self.assertFalse(result['guide_comparison_performed'])

    def test_media_metadata_is_fail_closed_and_sensitive_tags_are_rejected(self):
        self.assertTrue(media_tag_errors(Path('x.mp4'),None))
        with patch('tools.audit_public.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps({'format':{'tags':{'artist':'private'}}}))):
            self.assertIn('unexpected media metadata tag',media_tag_errors(Path('x.mp4'),'ffprobe'))
        with patch('tools.audit_public.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps({'format':{'tags':{'encoder':'Lavf'}}}))):
            self.assertEqual(media_tag_errors(Path('x.mp4'),'ffprobe'),[])

    def test_compressed_pixel_email_pattern_is_not_metadata(self):
        def box(kind,payload):return (8+len(payload)).to_bytes(4,'big')+kind+payload
        email=b'person'+b'@example.com'
        raw=box(b'ftyp',b'isom')+box(b'mdat',email)+box(b'moov',b'clean')
        self.assertFalse(binary_errors(media_metadata_bytes(raw,'.mp4')))
        raw=box(b'ftyp',b'isom')+box(b'mdat',b'pixels')+box(b'moov',email)
        self.assertIn('contact/email',binary_errors(media_metadata_bytes(raw,'.mp4')))
        path=b'/tmp/' + b'fixture.dat'
        self.assertFalse(binary_errors(path,compressed_payload=True))
        self.assertIn('absolute local path',binary_errors(path))
        gif=base64.b64decode('R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7')
        tagged=gif[:-1]+b'\x21\xfe'+bytes([len(email)])+email+b'\x00\x3b'
        self.assertIn('contact/email',binary_errors(media_metadata_bytes(tagged,'.gif')))
        split=gif[:-1]+b'\x21\xfe'+bytes([7])+email[:7]+bytes([len(email)-7])+email[7:]+b'\x00\x3b'
        self.assertIn('contact/email',binary_errors(media_metadata_bytes(split,'.gif')))
        self.assertFalse(binary_errors(media_metadata_bytes(gif,'.gif')))

    def test_media_root_and_total_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);media=root/'artifacts/media';media.mkdir(parents=True)
            for i in range(4):
                with (media/f'{i}.mp4').open('wb') as stream:stream.truncate(7_500_001)
            (root/'wrong.gif').write_bytes(b'GIF89a')
            with patch('tools.audit_public.media_tag_errors',return_value=[]), patch('tools.audit_public.binary_errors',return_value=[]):
                result=audit(root,ffprobe='ffprobe',require_checksums=False)
            self.assertIn('media total exceeds 30,000,000 bytes',result['errors'])
            self.assertTrue(any('media outside' in e for e in result['errors']))

    def test_required_media_missing_and_probe_failure_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); media=root/'artifacts/media';media.mkdir(parents=True)
            (media/'01_servo_no_collision.mp4').write_bytes(b'bad')
            with patch('tools.audit_public.media_tag_errors',return_value=['probe failed']):
                result=audit(root,ffprobe='missing',require_checksums=False,require_media=True)
            self.assertFalse(result['media_metadata_checked'])
            self.assertEqual(result['media_metadata_checked_count'],0)
            self.assertTrue(any('required four' in e for e in result['errors']))

    def test_invalid_utf8_is_fail_not_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'bad.py').write_bytes(b'\xff')
            guide=root/'guide_source';guide.write_bytes(b'\xff')
            (root/'artifacts').mkdir();(root/'artifacts/SHA256SUMS').write_bytes(b'\xff')
            result=audit(root,kit=root,guide=guide)
            self.assertEqual(result['status'],'FAIL')
            self.assertTrue(any('UTF-8' in e for e in result['errors']))


if __name__ == '__main__':
    unittest.main()
