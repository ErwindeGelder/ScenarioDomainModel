"""
Generate all relevant pdfs from .tex files.
"""

import subprocess
import os
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
    files = glob.glob(os.path.join(folder, '*.'+extension))
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
    tikzpath = os.path.join(folder, 'tikz')
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
            file.write('Processing file "{:s}"\n'.format(os.path.join(folder, texfile)))
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


def compile_doc(filename, git=None, other=None, newname=None, toggle=None, **kwargs):
    """ Compile a document.

    :param filename: The full name of the document, relative from the base folder of this git repo.
    :param git: The git commit version, if any other than the master.
    :param other: Any other files. Any file needs to be a tuple (<filename::str>, <args::dict>)
    :param newname: Rename the file, if needed.
    :param toggle: Set a toggle, if needed. Tuple: (<name of toggle::str>, <value::bool>).
    :param kwargs: Any additional arguments that will be used for the pdf function.
    """
    folder = os.path.dirname(os.path.splitext(filename)[0])
    filename = os.path.basename(os.path.splitext(filename)[0])
    if ARGS.overwrite is False:
        if newname is not None:
            if os.path.exists(os.path.join('..', folder, '{:s}.pdf'.format(newname))):
                clean_folder(os.path.join('..', folder))
                return
        else:
            if os.path.exists(os.path.join('..', folder, '{:s}.pdf'.format(filename))):
                clean_folder(os.path.join('..', folder))
                return

    if git is not None:
        call_output(['git', 'checkout', git])

    if other is not None:
        if not isinstance(other, List):
            other = [other]
        for file, arguments in other:
            compile_doc(file, **arguments)

    if toggle is not None:
        settoggle(os.path.join('..', folder, '{:s}.tex'.format(filename)), toggle[0], toggle[1])
    pdf(os.path.join('..', folder), filename, **kwargs)

    if newname is not None:
        copyfile(os.path.join('..', folder, '{:s}.pdf'.format(filename)),
                 os.path.join('..', folder, '{:s}.pdf'.format(newname)))

    # Reset toggle, otherwise cannot do git checkout
    if toggle is not None:
        settoggle(os.path.join('..', folder, '{:s}.tex'.format(filename)),
                  toggle[0], not toggle[1])

    if git is not None:
        call_output(['git', 'checkout', 'master'])


def compile_pr(i: int, **kwargs):
    """ Compile a progress report.

    :param i: The number of the progress report.
    :param kwargs: All arguments that are parsed to compile_doc.
    """
    filename = os.path.join('progress_reports', 'report{:02d}'.format(i),
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

    compile_doc(os.path.join('20171010 Summary', 'phd_summary'))

    # Do all the progress reports
    compile_doc(os.path.join('progress_reports', 'template', 'progress_report'))
    compile_pr(1, usebibtex=True)
    compile_pr(2, usebibtex=True)
    compile_pr(3, usebibtex=True,
               other=(os.path.join('20171111 IV2018 Ontology', 'root'), dict(usebibtex=True)))
    remove(os.path.join('..', '20171111 IV2018 Ontology', 'root.pdf'))  # Renamed to ontology
    compile_pr(4, usebibtex=True,
               other=(os.path.join('20171126 Parametrization', 'hyperparameter_selection'),
                      dict(usebibtex=True, toggle=('standalone', False))))
    compile_pr(5, usebiber=True,
               other=(os.path.join('20171111 IV2018 Ontology', 'root'), dict(usebiber=True)))
    remove(os.path.join('..', '20171111 IV2018 Ontology', 'root.pdf'))  # Renamed to ontology
    """    
    call_output(['git', 'checkout', 'PR6'])
    settoggle(os.path.join('..', '20180319 Completeness', 'completeness.tex'), 'standalone', False)
    pdf(os.path.join('..', '20180319 Completeness'), 'completeness', usebiber=True, log=False)
    pdf(os.path.join('..', 'progress_reports', 'report06'), 'progress_report_06', usebiber=True)
    call('git checkout ../"20180319 Completeness"/completeness.tex')
    call_output(['git', 'checkout', 'PR7'])
    pdf(os.path.join('..', 'progress_reports', 'report07'), 'progress_report_07', usebiber=True)
    call_output(['git', 'checkout', 'PR8'])
    pdf(os.path.join('..', '20180319 Completeness'), 'completeness', usebiber=True, log=False)
    pdf(os.path.join('..', 'progress_reports', 'report08'), 'progress_report_08', usebiber=True)
    call_output(['git', 'checkout', 'PR9'])
    pdf(os.path.join('..', '20180639 Journal paper ontology'), 'journal_ontology', usebiber=True,
        log=False)
    pdf(os.path.join('..', 'progress_reports', 'report09'), 'progress_report_09', usebiber=True)
    call_output(['git', 'checkout', 'PR10'])
    pdf(os.path.join('..', 'progress_reports', 'report10'), 'progress_report_10', usebiber=True)
    call_output(['git', 'checkout', 'PR11'])
    pdf(os.path.join('..', '20180917 GoNoGo'), 'GoNoGo', usebiber=True, log=False)
    pdf(os.path.join('..', 'progress_reports', 'report11'), 'progress_report_11')
    call_output(['git', 'checkout', 'PR12'])
    pdf(os.path.join('..', '20180924 Completeness paper'), 'completeness', usebiber=True,
        log=False)
    pdf(os.path.join('..', 'progress_reports', 'report12'), 'progress_report_12')
    call_output(['git', 'checkout', 'PR13'])
    pdf(os.path.join('..', '20180710 Ontology'), 'ontology', usebiber=True, log=False)
    pdf(os.path.join('..', 'progress_reports', 'report13'), 'progress_report_13', usebiber=True)
    call_output(['git', 'checkout', 'PR14'])
    pdf(os.path.join('..', '20180924 Completeness paper'), 'completeness', usebiber=True,
        log=False)
    pdf(os.path.join('..', '20181217 Completeness paper review'), 'cover_letter', log=False)
    pdf(os.path.join('..', 'progress_reports', 'report14'), 'progress_report_14', usebiber=True)
    call_output(['git', 'checkout', 'PR15'])
    pdf(os.path.join('..', '20180639 Journal paper ontology'), 'journal_ontology', usebiber=True,
        log=False)
    pdf(os.path.join('..', 'progress_reports', 'report15'), 'progress_report_15', usebiber=True)
    call_output(['git', 'checkout', 'PR16'])
    pdf(os.path.join('..', '20180639 Journal paper ontology'), 'journal_ontology', usebiber=True,
        log=False)
    pdf(os.path.join('..', 'progress_reports', 'report16'), 'progress_report_16', usebiber=True)
    call_output(['git', 'checkout', 'PR17'])
    pdf(os.path.join('..', '20180629 Journal paper ontology'), 'journal_ontology', usebiber=True,
        log=False)
    pdf(os.path.join('..', 'progress_reports', 'report17'), 'progress_report_17')
    call_output(['git', 'checkout', 'PR18'])
    pdf(os.path.join('..', '20190505 Assessment Strategy'), 'assessment_strategy', usebiber=True,
        log=False)
    pdf(os.path.join('..', 'progress_reports', 'report18'), 'progress_report_18')
    call_output(['git', 'checkout', 'PR19'])
    pdf(os.path.join('..', 'progress_reports', 'report19'), 'progress_report_19', usebiber=True)
    call_output(['git', 'checkout', 'PR20'])
    pdf(os.path.join('..', '20180629 Journal paper ontology'), 'journal_ontology', usebiber=True,
        log=False)
    pdf(os.path.join('..', 'progress_reports', 'report20'), 'progress_report_20', usebiber=True)
    call_output(['git', 'checkout', 'PR21'])
    pdf(os.path.join('..', '20190725 Scenario Risk'), 'scenariorisk', usebiber=True, log=False)
    pdf(os.path.join('..', 'progress_reports', 'report21'), 'progress_report_21', usebiber=True)

    # Revisions/versions
    call_output(['git', 'checkout', 'CompletenessPaperInit'])
    pdf(os.path.join('..', '20180924 Completeness paper'), 'completeness', usebiber=True,
        log=False)
    copyfile(os.path.join('..', '20180924 Completeness paper', 'completeness.pdf'),
             os.path.join('..', '20180924 Completeness paper',
                          '20181108 Completeness Initial.pdf'))

    call_output(['git', 'checkout', 'CompletenessPaperR1'])
    pdf(os.path.join('..', '20180924 Completeness paper'), 'completeness', usebiber=True,
        log=False)
    copyfile(os.path.join('..', '20180924 Completeness paper', 'completeness.pdf'),
             os.path.join('..', '20180924 Completeness paper',
                          '20181221 Completeness revision 1.pdf'))
    pdf(os.path.join('..', '20181217 Completeness paper review'), 'cover_letter')
    copyfile(os.path.join('..', '20181217 Completeness paper review', 'cover_letter.pdf'),
             os.path.join('..', '20181217 Completeness paper review',
                          '20181221 Cover letter revision 1.pdf'))

    call_output(['git', 'checkout', 'OntologyV1blue'])
    pdf(os.path.join('..', '20180629 Journal paper ontology'), 'journal_ontology', usebiber=True)
    copyfile(os.path.join('..', '20180629 Journal paper ontology', 'journal_ontology.pdf'),
             os.path.join('..', '20180629 Journal paper ontology', '20190525_V1_blue.pdf'))
    call_output(['git', 'checkout', 'OntologyV1'])
    pdf(os.path.join('..', '20180629 Journal paper ontology'), 'journal_ontology', usebiber=True)
    copyfile(os.path.join('..', '20180629 Journal paper ontology', 'journal_ontology.pdf'),
             os.path.join('..', '20180629 Journal paper ontology', '20190525_V1.pdf'))

    call_output(['git', 'checkout', 'OntologyV2limited'])
    pdf(os.path.join('..', '20180629 Journal paper ontology'), 'journal_ontology', usebiber=True)
    pdf(os.path.join('..', '20180629 Journal paper ontology'), 'journal_ontology_limited')
    copyfile(os.path.join('..', '20180629 Journal paper ontology', 'journal_ontology_limited.pdf'),
             os.path.join('..', '20180629 Journal paper ontology', '20190708_V2_limited.pdf'))
    call_output(['git', 'checkout', 'OntologyV2blue'])
    pdf(os.path.join('..', '20180629 Journal paper ontology'), 'journal_ontology', usebiber=True)
    copyfile(os.path.join('..', '20180629 Journal paper ontology', 'journal_ontology.pdf'),
             os.path.join('..', '20180629 Journal paper ontology', '20190708_V2_blue.pdf'))
    call_output(['git', 'checkout', 'OntologyV2'])
    pdf(os.path.join('..', '20180629 Journal paper ontology'), 'journal_ontology', usebiber=True)
    copyfile(os.path.join('..', '20180629 Journal paper ontology', 'journal_ontology.pdf'),
             os.path.join('..', '20180629 Journal paper ontology', '20190708_V2.pdf'))
    call_output(['git', 'checkout', 'OntologyV3'])
    pdf(os.path.join('..', '20180629 Journal paper ontology'), 'journal_ontology', usebiber=True)
    copyfile(os.path.join('..', '20180629 Journal paper ontology', 'journal_ontology.pdf'),
             os.path.join('..', '20180629 Journal paper ontology', '20190801_V3.pdf'))
    call_output(['git', 'checkout', 'OntologyV3blue'])
    pdf(os.path.join('..', '20180629 Journal paper ontology'), 'journal_ontology', usebiber=True)
    copyfile(os.path.join('..', '20180629 Journal paper ontology', 'journal_ontology.pdf'),
             os.path.join('..', '20180629 Journal paper ontology', '20190801_V3blue.pdf'))
    call_output(['git', 'checkout', 'OntologyV4blue'])
    pdf(os.path.join('..', '20180629 Journal paper ontology'), 'journal_ontology', usebiber=True)
    pdf(os.path.join('..', '20180629 Journal paper ontology'), 'journal_ontology_limited')
    copyfile(os.path.join('..', '20180629 Journal paper ontology', 'journal_ontology_limited.pdf'),
             os.path.join('..', '20180629 Journal paper ontology', '20190812_V4blue.pdf'))

    # All other stuff
    call_output(['git', 'checkout', 'master'])
    pdf(os.path.join('..', '20171111 IV2018 Ontology'), 'ontology', usebiber=True)
    pdf(os.path.join('..', '20180109 Ontology presentation'), 'ontology', usebibtex=True)
    pdf(os.path.join('..', '20180110 Test scenario generation presentation'),
        'scenario_generation', usebibtex=True)
    settoggle(os.path.join('..', '20171126 Parametrization', 'hyperparameter_selection.tex'),
              'standalone', True)
    pdf(os.path.join('..', '20171126 Parametrization'), 'hyperparameter_selection', usebibtex=True)
    settoggle(os.path.join('..', '20180207 Similarity', 'scenario_similarity.tex'), 'standalone',
              True)
    pdf(os.path.join('..', '20180207 Similarity'), 'scenario_similarity', usebiber=True)
    pdf(os.path.join('..', '20180320 Detailed scenario description'), 'scenario_description',
        usebiber=True)
    settoggle(os.path.join('..', '20180319 Completeness', 'completeness.tex'), 'standalone', True)
    pdf(os.path.join('..', '20180319 Completeness'), 'completeness', usebiber=True)
    pdf(os.path.join('..', '20180521 Summary GoNoGo'), 'phd_summary', usebiber=True)
    pdf(os.path.join('..', '20180629 Journal paper ontology'), 'journal_ontology', usebiber=True)
    pdf(os.path.join('..', '20180710 Ontology'), 'ontology', usebiber=True)
    pdf(os.path.join('..', '20180917 GoNoGo'), 'GoNoGo', usebiber=True)
    pdf(os.path.join('..', '20181002 Completeness question'), 'completeness_questions',
        usebiber=True)
    pdf(os.path.join('..', '20180924 Completeness paper'), 'completeness', usebiber=True)
    pdf(os.path.join('..', '20181217 Completeness paper review'), 'cover_letter')
    pdf(os.path.join('..', '20190505 Assessment Strategy'), 'assessment_strategy', usebiber=True)
    pdf(os.path.join('..', '20190725 Scenario Risk'), 'scenariorisk', usebiber=True)

    # Delete folder that has wrong name
    call('rm "{:s}" -r'.format(os.path.join('..', '20180639 Journal paper ontology')))
    """
