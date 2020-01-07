import subprocess
import time
import shutil
import signal
import os
from collections import namedtuple
import re
import sys

from testUtils import Utils

Wallet=namedtuple("Wallet", "name password host port")
# pylint: disable=too-many-instance-attributes
class WalletMgr(object):
    __walletLogOutFile="test_kbitconchd_out.log"
    __walletLogErrFile="test_kbitconchd_err.log"
    __walletDataDir="test_wallet_0"
    __MaxPort=9999

    # pylint: disable=too-many-arguments
    # walletd [True|False] True=Launch wallet(kbitconchd) process; False=Manage launch process externally.
    def __init__(self, walletd, nodebitconchPort=8888, nodebitconchHost="localhost", port=9899, host="localhost"):
        self.walletd=walletd
        self.nodebitconchPort=nodebitconchPort
        self.nodebitconchHost=nodebitconchHost
        self.port=port
        self.host=host
        self.wallets={}
        self.__walletPid=None

    def getWalletEndpointArgs(self):
        if not self.walletd or not self.isLaunched():
            return ""

        return " --wallet-url http://%s:%d" % (self.host, self.port)

    def getArgs(self):
        return " --url http://%s:%d%s %s" % (self.nodebitconchHost, self.nodebitconchPort, self.getWalletEndpointArgs(), Utils.MiscBitconchClientArgs)

    def isLaunched(self):
        return self.__walletPid is not None

    def isLocal(self):
        return self.host=="localhost" or self.host=="127.0.0.1"

    def findAvailablePort(self):
        for i in range(WalletMgr.__MaxPort):
            port=self.port+i
            if port > WalletMgr.__MaxPort:
                port-=WalletMgr.__MaxPort
            if Utils.arePortsAvailable(port):
                return port
            if Utils.Debug: Utils.Print("Port %d not available for %s" % (port, Utils.BitconchWalletPath))

        Utils.errorExit("Failed to find free port to use for %s" % (Utils.BitconchWalletPath))

    def launch(self):
        if not self.walletd:
            Utils.Print("ERROR: Wallet Manager wasn't configured to launch kbitconchd")
            return False

        if self.isLaunched():
            return True

        if self.isLocal():
            self.port=self.findAvailablePort()

        pgrepCmd=Utils.pgrepCmd(Utils.BitconchWalletName)
        if Utils.Debug:
            portTaken=False
            if self.isLocal():
                if not Utils.arePortsAvailable(self.port):
                    portTaken=True
            psOut=Utils.checkOutput(pgrepCmd.split(), ignoreError=True)
            if psOut or portTaken:
                statusMsg=""
                if psOut:
                    statusMsg+=" %s - {%s}." % (pgrepCmd, psOut)
                if portTaken:
                    statusMsg+=" port %d is NOT available." % (self.port)
                Utils.Print("Launching %s, note similar processes running. %s" % (Utils.BitconchWalletName, statusMsg))

        cmd="%s --data-dir %s --config-dir %s --http-server-address=%s:%d --verbose-http-errors" % (
            Utils.BitconchWalletPath, WalletMgr.__walletDataDir, WalletMgr.__walletDataDir, self.host, self.port)
        if Utils.Debug: Utils.Print("cmd: %s" % (cmd))
        with open(WalletMgr.__walletLogOutFile, 'w') as sout, open(WalletMgr.__walletLogErrFile, 'w') as serr:
            popen=subprocess.Popen(cmd.split(), stdout=sout, stderr=serr)
            self.__walletPid=popen.pid

        # Give kbitconchd time to warm up
        time.sleep(2)

        try:
            if Utils.Debug: Utils.Print("Checking if %s launched. %s" % (Utils.BitconchWalletName, pgrepCmd))
            psOut=Utils.checkOutput(pgrepCmd.split())
            if Utils.Debug: Utils.Print("Launched %s. {%s}" % (Utils.BitconchWalletName, psOut))
        except subprocess.CalledProcessError as ex:
            Utils.errorExit("Failed to launch the wallet manager")

        return True

    def create(self, name, accounts=None, exitOnError=True):
        wallet=self.wallets.get(name)
        if wallet is not None:
            if Utils.Debug: Utils.Print("Wallet \"%s\" already exists. Returning same." % name)
            return wallet
        p = re.compile(r'\n\"(\w+)\"\n', re.MULTILINE)
        cmdDesc="wallet create"
        cmd="%s %s %s --name %s --to-console" % (Utils.BitconchClientPath, self.getArgs(), cmdDesc, name)
        if Utils.Debug: Utils.Print("cmd: %s" % (cmd))
        retStr=None
        maxRetryCount=4
        retryCount=0
        while True:
            try:
                retStr=Utils.checkOutput(cmd.split())
                break
            except subprocess.CalledProcessError as ex:
                retryCount+=1
                if retryCount<maxRetryCount:
                    delay=10
                    pgrepCmd=Utils.pgrepCmd(Utils.BitconchWalletName)
                    psOut=Utils.checkOutput(pgrepCmd.split())
                    portStatus="N/A"
                    if self.isLocal():
                        if Utils.arePortsAvailable(self.port):
                            portStatus="AVAILABLE"
                        else:
                            portStatus="NOT AVAILABLE"
                    if Utils.Debug: Utils.Print("%s was not accepted, delaying for %d seconds and trying again. port %d is %s. %s - {%s}" % (cmdDesc, delay, self.port, pgrepCmd, psOut))
                    time.sleep(delay)
                    continue

                msg=ex.output.decode("utf-8")
                errorMsg="ERROR: Failed to create wallet - %s. %s" % (name, msg)
                if exitOnError:
                    Utils.errorExit("%s" % (errorMsg))
                Utils.Print("%s" % (errorMsg))
                return None

        m=p.search(retStr)
        if m is None:
            if exitOnError:
                Utils.cmdError("could not create wallet %s" % (name))
                Utils.errorExit("Failed  to create wallet %s" % (name))

            Utils.Print("ERROR: wallet password parser failure")
            return None
        p=m.group(1)
        wallet=Wallet(name, p, self.host, self.port)
        self.wallets[name] = wallet

        if accounts:
            self.importKeys(accounts,wallet)

        return wallet

    def importKeys(self, accounts, wallet, ignoreDupKeyWarning=False):
        for account in accounts:
            Utils.Print("Importing keys for account %s into wallet %s." % (account.name, wallet.name))
            if not self.importKey(account, wallet, ignoreDupKeyWarning):
                Utils.Print("ERROR: Failed to import key for account %s" % (account.name))
                return False

    def importKey(self, account, wallet, ignoreDupKeyWarning=False):
        warningMsg="Key already in wallet"
        cmd="%s %s wallet import --name %s --private-key %s" % (
            Utils.BitconchClientPath, self.getArgs(), wallet.name, account.ownerPrivateKey)
        if Utils.Debug: Utils.Print("cmd: %s" % (cmd))
        try:
            Utils.checkOutput(cmd.split())
        except subprocess.CalledProcessError as ex:
            msg=ex.output.decode("utf-8")
            if warningMsg in msg:
                if not ignoreDupKeyWarning:
                    Utils.Print("WARNING: This key is already imported into the wallet.")
            else:
                Utils.Print("ERROR: Failed to import account owner key %s. %s" % (account.ownerPrivateKey, msg))
                return False

        if account.activePrivateKey is None:
            Utils.Print("WARNING: Active private key is not defined for account \"%s\"" % (account.name))
        else:
            cmd="%s %s wallet import --name %s  --private-key %s" % (
                Utils.BitconchClientPath, self.getArgs(), wallet.name, account.activePrivateKey)
            if Utils.Debug: Utils.Print("cmd: %s" % (cmd))
            try:
                Utils.checkOutput(cmd.split())
            except subprocess.CalledProcessError as ex:
                msg=ex.output.decode("utf-8")
                if warningMsg in msg:
                    if not ignoreDupKeyWarning:
                        Utils.Print("WARNING: This key is already imported into the wallet.")
                else:
                    Utils.Print("ERROR: Failed to import account active key %s. %s" %
                                (account.activePrivateKey, msg))
                    return False

        return True

    def lockWallet(self, wallet):
        cmd="%s %s wallet lock --name %s" % (Utils.BitconchClientPath, self.getArgs(), wallet.name)
        if Utils.Debug: Utils.Print("cmd: %s" % (cmd))
        if 0 != subprocess.call(cmd.split(), stdout=Utils.FNull):
            Utils.Print("ERROR: Failed to lock wallet %s." % (wallet.name))
            return False

        return True

    def unlockWallet(self, wallet):
        cmd="%s %s wallet unlock --name %s" % (Utils.BitconchClientPath, self.getArgs(), wallet.name)
        if Utils.Debug: Utils.Print("cmd: %s" % (cmd))
        popen=subprocess.Popen(cmd.split(), stdout=Utils.FNull, stdin=subprocess.PIPE)
        _, errs = popen.communicate(input=wallet.password.encode("utf-8"))
        if 0 != popen.wait():
            Utils.Print("ERROR: Failed to unlock wallet %s: %s" % (wallet.name, errs.decode("utf-8")))
            return False

        return True

    def lockAllWallets(self):
        cmd="%s %s wallet lock_all" % (Utils.BitconchClientPath, self.getArgs())
        if Utils.Debug: Utils.Print("cmd: %s" % (cmd))
        if 0 != subprocess.call(cmd.split(), stdout=Utils.FNull):
            Utils.Print("ERROR: Failed to lock all wallets.")
            return False

        return True

    def getOpenWallets(self):
        wallets=[]

        p = re.compile(r'\s+\"(\w+)\s\*\",?\n', re.MULTILINE)
        cmd="%s %s wallet list" % (Utils.BitconchClientPath, self.getArgs())
        if Utils.Debug: Utils.Print("cmd: %s" % (cmd))
        retStr=None
        try:
            retStr=Utils.checkOutput(cmd.split())
        except subprocess.CalledProcessError as ex:
            msg=ex.output.decode("utf-8")
            Utils.Print("ERROR: Failed to open wallets. %s" % (msg))
            return False

        m=p.findall(retStr)
        if m is None:
            Utils.Print("ERROR: wallet list parser failure")
            return None
        wallets=m

        return wallets

    def getKeys(self, wallet):
        keys=[]

        p = re.compile(r'\n\s+\"(\w+)\"\n', re.MULTILINE)
        cmd="%s %s wallet private_keys --name %s --password %s " % (Utils.BitconchClientPath, self.getArgs(), wallet.name, wallet.password)
        if Utils.Debug: Utils.Print("cmd: %s" % (cmd))
        retStr=None
        try:
            retStr=Utils.checkOutput(cmd.split())
        except subprocess.CalledProcessError as ex:
            msg=ex.output.decode("utf-8")
            Utils.Print("ERROR: Failed to get keys. %s" % (msg))
            return False
        m=p.findall(retStr)
        if m is None:
            Utils.Print("ERROR: wallet private_keys parser failure")
            return None
        keys=m

        return keys


    def dumpErrorDetails(self):
        Utils.Print("=================================================================")
        if self.__walletPid is not None:
            Utils.Print("Contents of %s:" % (WalletMgr.__walletLogOutFile))
            Utils.Print("=================================================================")
            with open(WalletMgr.__walletLogOutFile, "r") as f:
                shutil.copyfileobj(f, sys.stdout)
            Utils.Print("Contents of %s:" % (WalletMgr.__walletLogErrFile))
            Utils.Print("=================================================================")
            with open(WalletMgr.__walletLogErrFile, "r") as f:
                shutil.copyfileobj(f, sys.stdout)

    def killall(self, allInstances=False):
        """Kill kbitconch instances. allInstances will kill all kbitconch instances running on the system."""
        if self.__walletPid:
            Utils.Print("Killing wallet manager process %d" % (self.__walletPid))
            os.kill(self.__walletPid, signal.SIGKILL)

        if allInstances:
            cmd="pkill -9 %s" % (Utils.BitconchWalletName)
            if Utils.Debug: Utils.Print("cmd: %s" % (cmd))
            subprocess.call(cmd.split())


    @staticmethod
    def cleanup():
        dataDir=WalletMgr.__walletDataDir
        if os.path.isdir(dataDir) and os.path.exists(dataDir):
            shutil.rmtree(WalletMgr.__walletDataDir)
