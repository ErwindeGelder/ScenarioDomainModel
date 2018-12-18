"""
Generate all relevant pdfs from .tex files.
"""

import subprocess
import os
import glob
import fileinput
from typing import List, Tuple


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
        exts = ['dpth', 'log', 'run.xml', 'md5', 'pdf']
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
                runpdflatex = True
                n_rerun_changebar += 1
                if n_rerun_changebar < 5:  # Limit reruns, because sometimes it goes on infinitely
                    print("RERUN BECAUSE CHANGEBAR CHANGED !!!")
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


if __name__ == '__main__':
    if os.path.exists('log2.txt'):
        os.remove("log2.txt")  # Empty log

    pdf(os.path.join('..', '20171010 Summary'), 'phd_summary')

    # Do all the progress reports
    pdf(os.path.join('..', 'progress_reports', 'template'), 'progress_report')
    call_output(['git', 'checkout', 'PR1'])
    pdf(os.path.join('..', 'progress_reports', 'report01'), 'progress_report_01', usebibtex=True)
    call_output(['git', 'checkout', 'PR2'])
    pdf(os.path.join('..', 'progress_reports', 'report02'), 'progress_report_02', usebibtex=True)
    call_output(['git', 'checkout', 'PR3'])
    pdf(os.path.join('..', '20171111 IV2018 Ontology'), 'root', usebibtex=True, log=False)
    pdf(os.path.join('..', 'progress_reports', 'report03'), 'progress_report_03', usebibtex=True)
    os.remove(os.path.join('..', '20171111 IV2018 Ontology', 'root.pdf'))  # Renamed to ontology
    call_output(['git', 'checkout', 'PR4'])
    settoggle(os.path.join('..', '20171126 Parametrization', 'hyperparameter_selection.tex'),
              'standalone', False)
    pdf(os.path.join('..', '20171126 Parametrization'), 'hyperparameter_selection',
        usebibtex=True, log=False)
    pdf(os.path.join('..', 'progress_reports', 'report04'), 'progress_report_04', usebibtex=True)
    settoggle(os.path.join('..', '20171126 Parametrization', 'hyperparameter_selection.tex'),
              'standalone', True)
    call_output(['git', 'checkout', 'PR5'])
    pdf(os.path.join('..', '20171111 IV2018 Ontology'), 'root', usebiber=True, log=False)
    settoggle(os.path.join('..', '20180207 Similarity', 'scenario_similarity.tex'),
              'standalone', False)
    pdf(os.path.join('..', '20180207 Similarity'), 'scenario_similarity', usebiber=True, log=False)
    pdf(os.path.join('..', 'progress_reports', 'report05'), 'progress_report_05', usebiber=True)
    os.remove(os.path.join('..', '20171111 IV2018 Ontology', 'root.pdf'))  # Renamed to ontology
    call('git checkout ../"20180207 Similarity"/scenario_similarity.tex')
    if os.path.exists('log.txt'):
        os.remove('log.txt')  # To prevent complaining that log.txt will be overwritten by checkout
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
    pdf(os.path.join('..', 'progress_reports', 'report13'), 'progress_report_13')
    call_output(['git', 'checkout', 'master'])

    # All other stuff
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
    pdf(os.path.join('..', '20180639 Journal paper ontology'), 'journal_ontology', usebiber=True)
    pdf(os.path.join('..', '20180710 Ontology'), 'ontology', usebiber=True)
    pdf(os.path.join('..', '20180917 GoNoGo'), 'GoNoGo', usebiber=True)
    pdf(os.path.join('..', '20181002 Completeness question'), 'completeness_questions',
        usebiber=True)
    pdf(os.path.join('..', '20180924 Completeness paper'), 'completeness', usebiber=True)
