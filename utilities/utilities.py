"""
Generate all relevant pdfs from .tex files.
"""

import subprocess
import os
from os.path import join
import glob
import fileinput
import argparse
from typing import List, Tuple
from shutil import copyfile


PARSER = argparse.ArgumentParser(description="Compile the documents")
PARSER.add_argument('--overwrite', help="Overwrite existings pdfs", action="store_true")
ARGS = PARSER.parse_args()


def print_line():
    """ Print a line
    """
    print("#####################################################################################")


def call(string: str, **kwargs) -> None:
    """ Call a command

    :param string: command to call.
    :param kwargs: any arguments (other than shell=True) added to subprocess.call().
    """
    print("Subprocess: {:s}".format(string))
    subprocess.call(string, shell=True, **kwargs)
    print_line()


def call_output(calllist: List[str], **kwargs) -> str:
    """ Call a command and return output

    :param calllist: List of commands to call. Any options to the call should be provided as a
        seperate item of the list.
    :param kwargs: any arguments (other than shell=True) added to subprocess.call().
    :return: the output of the call.
    """
    print("Subprocess: {:s}".format(" ".join(calllist)))
    out = subprocess.check_output(calllist, shell=True, **kwargs)
    print_line()
    return out


def remove_files(extension: str, folder: str) -> None:
    """ Remove all files with the specified extension in a folder.

    :param extension: The extionsion of the file to be removed.
    :param folder: The folder in which the files should be removed.
    """
    files = glob.glob(join(folder, '*.'+extension))
    for file in files:
        print('Remove file "{:s}"'.format(file))
        os.remove(file)


def clean_folder(folder: str) -> None:
    """ Remove all kind of files in a folder.

    The following files are removed: '.aux', '.auxlock', '.bbl', '.bcf', '.blg', '.cb', '.cb2',
    '.hst', '.log', '.nav', '.out', '.run.xml', '.snm', '.synctex.gz', '.toc', '.ver'

    :param folder: Folder in which the files need to be removed.
    """
    # Remove files that might be generated when compiling a .tex file
    exts = ['aux', 'auxlock', 'bbl', 'bcf', 'blg', 'cb', 'cb2', 'hst', 'log', 'nav', 'out',
            'run.xml', 'snm', 'synctex.gz', 'toc', 'ver']
    for ext in exts:
        remove_files(ext, folder)

    # Remove contents of folder (if it is generated)
    tikzpath = join(folder, 'tikz')
    if os.path.exists(tikzpath):
        exts = ['cb', 'cb2', 'dpth', 'log', 'run.xml', 'md5', 'pdf']
        for ext in exts:
            remove_files(ext, tikzpath)


def settoggle(file: str, toggle: str, value: bool = True) -> None:
    """ Changes a toggle in a texfile.

    :param file: .tex file.
    :param toggle: Name of the toggle.
    :param value: True or False
    """
    for line in fileinput.input(file, inplace=True):
        if line[:13+len(toggle)] == '\\toggletrue{{{:s}}}'.format(toggle) or \
                        line[:14 + len(toggle)] == '\\togglefalse{{{:s}}}'.format(toggle):
            print("\\toggle{:s}{{{:s}}}".format("true" if value else "false", toggle))
        else:
            print(line, end="")


def pdf_latex(folder: str, texfile: str, output: bool = False) -> str:
    """ Run the pdflatex command

    :param folder: Folder of the texfile.
    :param texfile: Name of the texfile.
    :param output: Whether to return the output or not.
    """
    if output is False:
        cmd = 'pdflatex.exe -synctex=1 -interaction=nonstopmode -shell-escape "{:s}".tex'.\
            format(texfile)
        call(cmd, cwd=folder)
        return ''

    out = call_output(['pdflatex.exe', '-synctex=1', '-interaction=nonstopmode',
                       '-shell-escape', '"{:s}".tex'.format(texfile)],
                      cwd=folder)
    lines = out.decode('utf-8', 'ignore').split('\r\n')  # Ignore errors
    for line in lines:
        print(line)
    return lines


def bibtex(folder: str, texfile: str) -> None:
    """ Run the bibtex command

    :param folder: Folder of the texfile.
    :param texfile: Name of the texfile.
    """
    cmd = 'bibtex.exe "{:s}"'.format(texfile)
    call(cmd, cwd=folder)


def biber(folder: str, texfile: str) -> None:
    """ Run the biber command

    :param folder: Folder of the texfile.
    :param texfile: Name of the texfile.
    """
    cmd = 'biber.exe "{:s}"'.format(texfile)
    call(cmd, cwd=folder)


def pdf_till_no_rerun_warning(folder: str, texfile: str) -> Tuple[List, List]:
    """ Run PDF latex until no RERUN warning is returned.

    :param folder: Folder of the texfile.
    :param texfile: Name of the texfile.
    :return: A list of warnings and a list of badboxes
    """
    runpdflatex = True
    badboxes = []
    warnings = []
    n_rerun_changebar = 0
    while runpdflatex:
        badboxes = []
        warnings = []
        out = pdf_latex(folder, texfile, output=True)
        for i, line in enumerate(out):
            if line.find("Warning:") >= 0:
                warning = [line]
                i += 1
                while not out[i] == "":
                    warning.append(out[i])
                    i += 1
                warnings.append(warning)
            if len(line) >= 9:
                if line[:8] == "Overfull" or line[:9] == "Underfull":
                    badboxes.append([line, out[i+1]])
        runpdflatex = False  # Check if labels changed --> if so, rerun
        for warning in warnings:
            if warning[0].find('Label(s) may have changed') >= 0:
                runpdflatex = True
                print("RERUN BECAUSE LABELS CHANGED !!!")
                break
            if warning[0].find('Changebar info has changed.') >= 0:
                n_rerun_changebar += 1
                if n_rerun_changebar < 5:  # Limit reruns, because sometimes it goes on infinitely
                    runpdflatex = True
                    print("RERUN BECAUSE CHANGEBAR CHANGED !!! ({:d})".format(n_rerun_changebar))
                    break
    return warnings, badboxes


def pdf(folder: str, texfile: str, usebibtex: bool = False, usebiber: bool = False,
        log: bool = True) -> None:
    """ Create pdf from LaTeX file.

    :param folder: Folder of the texfile.
    :param texfile: Name of the texfile.
    :param usebibtex: Whether to use bibtex or not.
    :param usebiber: Whether to use biber or not.
    :param log: Whether to add comments to the log file or not.
    """
    clean_folder(folder)
    pdf_latex(folder, texfile)
    if usebibtex:
        bibtex(folder, texfile)
        pdf_latex(folder, texfile)
    if usebiber:
        biber(folder, texfile)
        pdf_latex(folder, texfile)
    warnings, badboxes = pdf_till_no_rerun_warning(folder, texfile)
    if log:
        with open("log2.txt", "a") as file:
            file.write("####################################################################\n")
            file.write('Processing file "{:s}"\n'.format(join(folder, texfile)))
            file.write("{:d} warnings\n".format(len(warnings)))
            for i, warning in enumerate(warnings):
                file.write("{:2d}: {:s}\n".format(i+1, warning[0]))
                for line in warning[1:]:
                    file.write("    {:s}\n".format(line))
            file.write("{:d} badboxes\n".format(len(badboxes)))
            for i, warning in enumerate(badboxes):
                file.write("{:2d}: {:s}\n".format(i+1, warning[0]))
                for line in warning[1:]:
                    file.write("    {:s}\n".format(line))
            file.write("\n")

    clean_folder(folder)


def compile_doc(filename, git=None, other=None, newname=None, toggle=None, add2pdfs=True,
                overwrite=ARGS.overwrite, **kwargs):
    """ Compile a document.

    :param filename: The full name of the document, relative from the base folder of this git repo.
    :param git: The git commit version, if any other than the master.
    :param other: Any other files. Any file needs to be a tuple (<filename::str>, <args::dict>)
    :param newname: Rename the file, if needed.
    :param toggle: Set a toggle, if needed. Tuple: (<name of toggle::str>, <value::bool>).
    :param overwrite: Whether to overwrite a file.
    :param kwargs: Any additional arguments that will be used for the pdf function.
    """
    folder = os.path.dirname(os.path.splitext(filename)[0])
    filename = os.path.basename(os.path.splitext(filename)[0])
    if overwrite is False:
        pdfname = '{:s}.pdf'.format(newname if newname is not None else filename)
        if os.path.exists(join('..', folder, pdfname)):
            clean_folder(join('..', folder))

            if not os.path.exists(join('..', 'pdfs', pdfname)) and add2pdfs:
                copyfile(join('..', folder, pdfname), join('..', 'pdfs', pdfname))
            return

    if git is not None:
        call_output(['git', 'checkout', git])

    if other is not None:
        if not isinstance(other, List):
            other = [other]
        for file, arguments in other:
            arguments.update(dict(log=False, overwrite=True, add2pdfs=False))
            compile_doc(file, **arguments)

    if toggle is not None:
        settoggle(join('..', folder, '{:s}.tex'.format(filename)), toggle[0], toggle[1])
    pdf(join('..', folder), filename, **kwargs)

    if other is not None:
        for file, _ in other:
            # Remove the generated files, because they need to be recompiled later.
            os.remove(join('..', '{:s}.pdf'.format(file)))

    if newname is not None:
        copyfile(join('..', folder, '{:s}.pdf'.format(filename)),
                 join('..', folder, '{:s}.pdf'.format(newname)))
        os.remove(join('..', folder, '{:s}.pdf'.format(filename)))
        if add2pdfs:
            copyfile(join('..', folder, '{:s}.pdf'.format(newname)),
                     join('..', 'pdfs', '{:s}.pdf'.format(newname)))
    elif add2pdfs:
        copyfile(join('..', folder, '{:s}.pdf'.format(filename)),
                 join('..', 'pdfs', '{:s}.pdf'.format(filename)))

    # Reset toggle, otherwise cannot do git checkout
    if toggle is not None:
        settoggle(join('..', folder, '{:s}.tex'.format(filename)),
                  toggle[0], not toggle[1])

    if git is not None:
        call_output(['git', 'checkout', 'master'])


def compile_pr(i: int, **kwargs):
    """ Compile a progress report.

    :param i: The number of the progress report.
    :param kwargs: All arguments that are parsed to compile_doc.
    """
    filename = join('progress_reports', 'report{:02d}'.format(i),
                    'progress_report_{:02d}'.format(i))
    kwargs.update(dict(git='PR{:d}'.format(i)))
    compile_doc(filename, **kwargs)


def remove(filename):
    """ Removes file after checking if it exists.

    :param filename:
    """
    if os.path.exists(filename):
        os.remove(filename)


if __name__ == '__main__':
    if os.path.exists('log2.txt'):
        if os.path.exists('log.txt'):
            os.remove('log.txt')  # Prevent complain that log.txt will be overwritten by checkout.
        os.remove("log2.txt")  # Empty log
    if not os.path.exists(join('..', 'pdfs')):
        os.mkdir(join('..', 'pdfs'))

    compile_doc(join('20171010 Summary', 'phd_summary'))

    # Do all the progress reports
    compile_doc(join('progress_reports', 'template', 'progress_report'))
    compile_pr(1, usebibtex=True)
    compile_pr(2, usebibtex=True)
    compile_pr(3, usebibtex=True,
               other=(join('20171111 IV2018 Ontology', 'root'), dict(usebibtex=True)))
    compile_pr(4, usebibtex=True,
               other=(join('20171126 Parametrization', 'hyperparameter_selection'),
                      dict(usebibtex=True, toggle=('standalone', False))))
    compile_pr(5, usebiber=True,
               other=[(join('20171111 IV2018 Ontology', 'root'), dict(usebiber=True)),
                      (join('20180207 Similarity', 'scenario_similarity'), dict(usebiber=True))])
    compile_pr(6, usebiber=True,
               other=(join('20180319 Completeness', 'completeness'), dict(usebiber=True)))
    compile_pr(7, usebiber=True)
    compile_pr(8, usebiber=True,
               other=(join('20180319 Completeness', 'completeness'), dict(usebiber=True)))
    compile_pr(9, usebiber=True,
               other=(join('20180639 Journal paper ontology', 'journal_ontology'),
                      dict(usebiber=True)))
    compile_pr(10, usebiber=True)
    compile_pr(11, other=(join('20180917 GoNoGo', 'GoNoGo'), dict(usebiber=True)))
    compile_pr(12,
               other=(join('20180924 Completeness paper', 'completeness'), dict(usebiber=True)))
    compile_pr(13, usebiber=True,
               other=(join('20180710 Ontology', 'ontology'), dict(usebiber=True)))
    compile_pr(14, usebiber=14,
               other=[(join('20180924 Completeness paper', 'completeness'), dict(usebiber=True)),
                      (join('20181217 Completeness paper review', 'cover_letter'), dict())])
    compile_pr(15, usebiber=True,
               other=(join('20180639 Journal paper ontology', 'journal_ontology'),
                      dict(usebiber=True)))
    compile_pr(16, usebiber=True,
               other=(join('20180639 Journal paper ontology', 'journal_ontology'),
                      dict(usebiber=True)))
    compile_pr(17, other=(join('20180629 Journal paper ontology', 'journal_ontology'),
                          dict(usebiber=True)))
    compile_pr(18, other=(join('20190505 Assessment Strategy', 'assessment_strategy'),
                          dict(usebiber=True)))
    compile_pr(19, usebiber=True)
    compile_pr(20, usebiber=True,
               other=(join('20180629 Journal paper ontology', 'journal_ontology'),
                      dict(usebiber=True)))
    compile_pr(21, usebiber=True,
               other=(join('20190725 Scenario Risk', 'scenariorisk'), dict(usebiber=True)))
    compile_pr(22, usebiber=True)
    compile_pr(23, usebiber=True, other=(join('20191004 Ontology revision letter',
                                              'ontology_revision'), dict(usebiber=True)))

    # Revisions/versions
    compile_doc(join('20180924 Completeness paper', 'completeness'), git='CompletenessPaperInit',
                newname='20181108 Completeness Initial', usebiber=True, log=False)
    compile_doc(join('20180924 Completeness paper', 'completeness'), git='CompletenessPaperR1',
                newname='20181221 Completeness revision 1', usebiber=True, log=False)
    compile_doc(join('20181217 Completeness paper review', 'cover_letter'),
                git='CompletenessPaperR1', newname='20181221 Cover letter revision 1')

    compile_doc(join('20180629 Journal paper ontology', 'journal_ontology'),
                git='OntologyV1blue', newname='20190525_V1_blue', usebiber=True, log=False)
    compile_doc(join('20180629 Journal paper ontology', 'journal_ontology'),
                git='OntologyV1', newname='20190525_V1', usebiber=True, log=False)
    compile_doc(join('20180629 Journal paper ontology', 'journal_ontology_limited'),
                git='OntologyV2limited', newname='20190708_V2_limited', log=False,
                other=(join('20180629 Journal paper ontology', 'journal_ontology'),
                       dict(usebiber=True)))
    compile_doc(join('20180629 Journal paper ontology', 'journal_ontology'),
                git='OntologyV2blue', newname='20190708_V2_blue', usebiber=True, log=False)
    compile_doc(join('20180629 Journal paper ontology', 'journal_ontology'),
                git='OntologyV2', newname='20190708_V2', usebiber=True, log=False)
    compile_doc(join('20180629 Journal paper ontology', 'journal_ontology'),
                git='OntologyV3', newname='20190801_V3', usebiber=True, log=False)
    compile_doc(join('20180629 Journal paper ontology', 'journal_ontology'),
                git='OntologyV3blue', newname='20190801_V3blue', usebiber=True, log=False)
    compile_doc(join('20180629 Journal paper ontology', 'journal_ontology_limited'),
                git='OntologyV4blue', newname='20190813_V4blue', log=False,
                other=(join('20180629 Journal paper ontology', 'journal_ontology'),
                       dict(usebiber=True)))
    compile_doc(join('20180629 Journal paper ontology', 'journal_ontology_limited'),
                git='OntologyV4lessblue', newname='20190813_V4lessblue', log=False,
                other=(join('20180629 Journal paper ontology', 'journal_ontology'),
                       dict(usebiber=True)))
    compile_doc(join('20180629 Journal paper ontology', 'journal_ontology_limited'),
                git='OntologyV4', newname='20190813_V4', log=False,
                other=(join('20180629 Journal paper ontology', 'journal_ontology'),
                       dict(usebiber=True)))
    compile_doc(join('20180629 Journal paper ontology', 'journal_ontology'), log=False,
                git='OntologyV5submitted', newname='20190816_OntologySubmitted', usebiber=True)
    compile_doc(join('20180629 Journal paper ontology', 'journal_ontology'), log=False,
                git='OntologyR1', newname='20190820_OntologyR1', usebiber=True)
    compile_doc(join('20190819 Journal paper ontology cover', 'ontology_cover'), log=False,
                git='OntologyR1', newname='20190820_OntologyCoverR1', usebiber=True)

    # All other stuff
    compile_doc(join('20171111 IV2018 Ontology', 'ontology'), usebiber=True)
    compile_doc(join('20180109 Ontology presentation', 'ontology'), usebibtex=True)
    compile_doc(join('20180110 Test scenario generation presentation', 'scenario_generation'),
                usebibtex=True)
    compile_doc(join('20171126 Parametrization', 'hyperparameter_selection'), usebibtex=True)
    compile_doc(join('20180207 Similarity', 'scenario_similarity'), usebiber=True)
    compile_doc(join('20180320 Detailed scenario description', 'scenario_description'),
                usebiber=True)
    compile_doc(join('20180319 Completeness', 'completeness'), usebiber=True)
    compile_doc(join('20180521 Summary GoNoGo', 'phd_summary'), usebiber=True)
    compile_doc(join('20180629 Journal paper ontology', 'journal_ontology'), usebiber=True)
    compile_doc(join('20180710 Ontology', 'ontology'), usebiber=True)
    compile_doc(join('20180917 GoNoGo', 'GoNoGo'), usebiber=True)
    compile_doc(join('20181002 Completeness question', 'completeness_questions'), usebiber=True)
    compile_doc(join('20180924 Completeness paper', 'completeness'), usebiber=True)
    compile_doc(join('20181217 Completeness paper review', 'cover_letter'))
    compile_doc(join('20190505 Assessment Strategy', 'assessment_strategy'), usebiber=True)
    compile_doc(join('20190725 Scenario Risk', 'scenariorisk'), usebiber=True)
    compile_doc(join('20190819 Journal paper ontology cover', 'ontology_cover'), usebiber=True)
    compile_doc(join('20191004 Ontology revision letter', 'ontology_revision'), usebiber=True)

    # Delete folder that has wrong name
    if os.path.exists(join('..', '20180639 Journal paper ontology')):
        call('rm "{:s}" -r'.format(join('..', '20180639 Journal paper ontology')))
