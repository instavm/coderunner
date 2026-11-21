"""Host backend safety and lifecycle tests, without requiring macOS or a VM image."""
import argparse
import json
import subprocess
import tempfile
import unittest
import urllib.error
from unittest import mock

import macos_vm


class MacOSVMTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = macos_vm.parser().parse_args(['--state-dir', self.temp.name, 'start'])
        self.vm = macos_vm.MacOSVM(self.args)
        self.addCleanup(lambda: self.vm.control_dir.rmdir() if self.vm.control_dir.exists() else None)
        macos_vm.private_write(self.vm.manifest, json.dumps({
            'name': self.args.name, 'user': 'admin', 'port': 8223, 'network': 'nat',
        }))

    def test_name_rejects_paths_and_shell_syntax(self):
        for name in ('../victim', '-option', 'a/b', 'bad;cmd', 'two words', ''):
            with self.subTest(name=name), self.assertRaises(argparse.ArgumentTypeError):
                macos_vm.validate_name(name)

    def test_private_state_and_vm_scoped_known_hosts(self):
        macos_vm.private_write(self.vm.token, 'secret')
        self.assertEqual(self.vm.token.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.vm.state.stat().st_mode & 0o777, 0o700)
        args = self.vm.ssh_args('192.168.64.2')
        self.assertIn('StrictHostKeyChecking=accept-new', args)
        self.assertIn(f'UserKnownHostsFile={self.vm.state / "known_hosts"}', args)
        self.assertIn('BatchMode=yes', args)
        self.assertEqual(self.vm.env['TART_NO_AUTO_PRUNE'], '1')
        self.assertLess(len(str(self.vm.control).encode()) + 17, 104)

    def test_create_does_not_download_when_disk_is_low(self):
        self.vm.manifest.unlink()
        self.args = macos_vm.parser().parse_args(['--state-dir', self.temp.name, 'create'])
        vm = macos_vm.MacOSVM(self.args)
        with mock.patch('macos_vm.run', return_value=mock.Mock(stdout=str(16 * 2**30))), \
                mock.patch('macos_vm.shutil.disk_usage', return_value=mock.Mock(free=2**30)), \
                mock.patch.object(vm, 'tart_run') as tart, self.assertRaisesRegex(RuntimeError, '50 GiB'):
            vm.create()
        tart.assert_not_called()

    def test_new_guest_cannot_provision_offline(self):
        self.args.network = 'none'
        with mock.patch.object(self.vm, 'running') as running, self.assertRaisesRegex(RuntimeError, 'internet'):
            self.vm.start()
        running.assert_not_called()

    def test_running_guest_cannot_silently_change_network(self):
        (self.vm.state / 'provisioned').touch()
        self.args.network = 'none'
        with mock.patch.object(self.vm, 'running', return_value=True), \
                self.assertRaisesRegex(RuntimeError, 'Stop the VM'):
            self.vm.start()

    def test_running_state_reads_tart_json(self):
        output = subprocess.CompletedProcess([], 0, stdout='[{"Name":"coderunner-macos","State":"running"}]')
        with mock.patch.object(self.vm, 'tart_run', return_value=output):
            self.assertTrue(self.vm.running())

    def test_tunnel_only_forwards_loopback_and_checks_bind(self):
        with mock.patch('macos_vm.run') as run:
            self.vm.tunnel('192.168.64.2')
        args = run.call_args.args[0]
        self.assertIn('127.0.0.1:8223:127.0.0.1:8222', args)
        self.assertIn('ExitOnForwardFailure=yes', args)

    def test_ssh_readiness_refreshes_a_stale_dhcp_address(self):
        with mock.patch('macos_vm.socket.create_connection', side_effect=[OSError(), mock.MagicMock()]) as connect, \
                mock.patch.object(self.vm, 'tart_run', return_value=mock.Mock(stdout='192.168.128.2\n')), \
                mock.patch('macos_vm.time.sleep'):
            self.assertEqual(self.vm.wait_ssh('192.168.64.2'), '192.168.128.2')
        self.assertEqual(connect.call_args.args[0], ('192.168.128.2', 22))

    def test_smoke_requires_unauthorized_rejection(self):
        with mock.patch.object(self.vm, 'request', return_value={}), \
                self.assertRaisesRegex(RuntimeError, 'Unauthenticated'):
            self.vm.smoke()

    def test_smoke_accepts_the_existing_rest_response_schema(self):
        unauthorized = urllib.error.HTTPError('http://localhost', 401, '', {}, None)
        result = {'stdout': '{"system": "Darwin", "hypervisor": "1"}', 'stderr': '', 'execution_time': 1}
        with mock.patch.object(self.vm, 'request', side_effect=[unauthorized, result]), mock.patch('builtins.print'):
            self.vm.smoke()


if __name__ == '__main__':
    unittest.main()
