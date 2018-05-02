import subprocess
import os
import glob
import fileinput


def print_line():
    print("#####################################################################################")


def call(string, **kwargs):
    print("Subprocess: {:s}".format(string))
    subprocess.call(string, shell=True, **kwargs)
    print_line()


def call_output(calllist, **kwargs):
    print("Subprocess: {:s}".format(" ".join(calllist)))
    out = subprocess.check_output(calllist, shell=True, **kwargs)
    print_line()
    return out


def remove_files(extension, folder):
    files = glob.glob(os.path.join(folder, '*.'+extension))
    for file in files:
        print('Remove file "{:s}"'.format(file))
        os.remove(file)


def clean_folder(folder):
    # Remove files that might be generated when compiling a .tex file
    exts = ['aux', 'auxlock', 'bbl', 'bcf', 'blg', 'hst', 'log', 'nav', 'out', 'run.xml', 'snm', 'synctex.gz', 'toc',
            'ver']
    for ext in exts:
        remove_files(ext, folder)

    # Remove contents of folder (if it is generated)
    tikzpath = os.path.join(folder, 'tikz')
    if os.path.exists(tikzpath):
        exts = ['dpth', 'log', 'run.xml', 'md5', 'pdf']
        for ext in exts:
            remove_files(ext, tikzpath)


def settoggle(file, toggle, value=True):
    for line in fileinput.input(file, inplace=True):
        if line[:13+len(toggle)] == '\\toggletrue{{{:s}}}'.format(toggle) or \
                        line[:14 + len(toggle)] == '\\togglefalse{{{:s}}}'.format(toggle):
            print("\\toggle{:s}{{{:s}}}".format("true" if value else "false", toggle))
        else:
            print(line, end="")


def pdf_latex(folder, texfile, output=False):
    if output is False:
        cmd = 'pdflatex.exe -synctex=1 -interaction=nonstopmode -shell-escape "{:s}".tex'.format(texfile)
        call(cmd, cwd=folder)
    else:
        out = call_output(['pdflatex.exe', '-synctex=1', '-interaction=nonstopmode', '-shell-escape',
                           '"{:s}".tex'.format(texfile)],
                          cwd=folder)
        lines = out.decode().split('\r\n')
        for line in lines:
            print(line)
        return lines


def bibtex(folder, texfile):
    cmd = 'bibtex.exe "{:s}"'.format(texfile)
    call(cmd, cwd=folder)


def biber(folder, texfile):
    cmd = 'biber.exe "{:s}"'.format(texfile)
    call(cmd, cwd=folder)


def pdf(folder, texfile, usebibtex=False, usebiber=False, clean=True, log=True):
    if clean:
        clean_folder(folder)
    pdf_latex(folder, texfile)
    if usebibtex:
        bibtex(folder, texfile)
        pdf_latex(folder, texfile)
    if usebiber:
        biber(folder, texfile)
        pdf_latex(folder, texfile)
    runpdflatex = True
    badboxes = []
    warning = []
    while runpdflatex:
        badboxes = []
        warning = []
        out = pdf_latex(folder, texfile, output=True)
        for i in range(len(out)):
            line = out[i]
            if line.find("Warning:") >= 0:
                w = [line]
                i += 1
                while not out[i] == "":
                    w.append(out[i])
                    i += 1
                warning.append(w)
            if len(line) >= 9:
                if line[:8] == "Overfull" or line[:9] == "Underfull":
                    badboxes.append([line, out[i+1]])
        runpdflatex = False  # Check if labels changed --> if so, rerun
        for w in warning:
            if w[0].find('Label(s) may have changed') >= 0:
                runpdflatex = True
                print("RERUN BECAUSE LABELS CHANGED !!!")
                break
    if log:
        with open("log.txt", "a") as f:
            f.write("####################################################################\n")
            f.write('Processing file "{:s}"\n'.format(os.path.join(folder, texfile)))
            f.write("{:d} warnings\n".format(len(warning)))
            for i, w in enumerate(warning):
                f.write("{:2d}: {:s}\n".format(i+1, w[0]))
                for l in w[1:]:
                    f.write("    {:s}\n".format(l))
            f.write("{:d} badboxes\n".format(len(badboxes)))
            for i, w in enumerate(badboxes):
                f.write("{:2d}: {:s}\n".format(i+1, w[0]))
                for l in w[1:]:
                    f.write("    {:s}\n".format(l))
            f.write("\n")

    if clean:
        clean_folder(folder)


if __name__ == '__main__':
    if os.path.exists('log.txt'):
        os.remove("log.txt")  # Empty log
    pdf(os.path.join('..', '20171010 Summary'), 'phd_summary')

    # Do all the progress reports
    pdf(os.path.join('..', 'progress_reports', 'template'), 'progress_report')
    call_output('git checkout PR1')
    pdf(os.path.join('..', 'progress_reports', 'report01'), 'progress_report_01', usebibtex=True)
    call_output('git checkout PR2')
    pdf(os.path.join('..', 'progress_reports', 'report02'), 'progress_report_02', usebibtex=True)
    call_output('git checkout PR3')
    pdf(os.path.join('..', 'progress_reports', 'report03'), 'progress_report_03', usebibtex=True)
    call_output('git checkout PR4')
    settoggle(os.path.join('..', '20171126 Parametrization', 'hyperparameter_selection.tex'), 'standalone', False)
    pdf(os.path.join('..', '20171126 Parametrization'), 'hyperparameter_selection', usebibtex=True, log=False)
    pdf(os.path.join('..', 'progress_reports', 'report04'), 'progress_report_04', usebibtex=True)
    call_output('git checkout PR5')
    settoggle(os.path.join('..', '20180207 Similarity', 'scenario_similarity.tex'), 'standalone', False)
    pdf(os.path.join('..', '20180207 Similarity'), 'scenario_similarity', usebiber=True, log=False)
    pdf(os.path.join('..', 'progress_reports', 'report05'), 'progress_report_05', usebiber=True)
    call_output('git checkout PR6')
    settoggle(os.path.join('..', '20180319 Completeness', 'completeness.tex'), 'standalone', False)
    pdf(os.path.join('..', '20180319 Completeness'), 'completeness', usebiber=True, log=False)
    pdf(os.path.join('..', 'progress_reports', 'report06'), 'progress_report_06', usebiber=True)
    call_output('git checkout master')

    # All other stuff
    pdf(os.path.join('..', '20171111 IV2018 Ontology'), 'ontology', usebiber=True)
    pdf(os.path.join('..', '20180109 Ontology presentation'), 'ontology', usebibtex=True)
    pdf(os.path.join('..', '20180110 Test scenario generation presentation'), 'scenario_generation', usebibtex=True)
    settoggle(os.path.join('..', '20171126 Parametrization', 'hyperparameter_selection.tex'), 'standalone', True)
    pdf(os.path.join('..', '20171126 Parametrization'), 'hyperparameter_selection', usebibtex=True)
    settoggle(os.path.join('..', '20180207 Similarity', 'scenario_similarity.tex'), 'standalone', True)
    pdf(os.path.join('..', '20180207 Similarity'), 'scenario_similarity', usebiber=True)
    pdf(os.path.join('..', '20180320 Detailed scenario description'), 'scenario_description', usebiber=True)
    settoggle(os.path.join('..', '20180319 Completeness', 'completeness.tex'), 'standalone', True)
    pdf(os.path.join('..', '20180319 Completeness'), 'completeness', usebiber=True, log=True)
