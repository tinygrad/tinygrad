from extra.sass.assembler.CuInsAssemblerRepos import CuInsAssemblerRepos
from extra.sass.assembler.CuInsFeeder import CuInsFeeder

import argparse
parser = argparse.ArgumentParser(usage="""
Run the instruction solver
""")
parser.add_argument("-s", "--src", required=True, help="input sass file path")
parser.add_argument("-d", "--dst", required=True, help="solved repo path (txt)")
parser.add_argument("-a", "--arch", required=True, help="SM architecture (SM_80, SM_89)")
args = parser.parse_args()

sassname, savname, arch = args.src, args.dst, args.arch


def constructReposFromFile(sassname, savname, arch):
    feeder = CuInsFeeder(sassname, archfilter=arch)
    repos = CuInsAssemblerRepos(arch=arch)
    repos.update(feeder)
    feeder.restart()
    repos.save2file(savname)
    repos.verify(feeder)
    return repos

def verifyReposFromFile(sassname, reposfile, arch='sm_75'):

    # initialize a feeder with sass
    feeder = CuInsFeeder(sassname, archfilter=arch)

    # initialize an empty repos
    repos = CuInsAssemblerRepos(reposfile, arch=arch)#

    # verify the repos
    repos.verify(feeder)

if __name__ == '__main__':
    constructReposFromFile(sassname, savname, arch=arch)
    print('### Construction done!')
