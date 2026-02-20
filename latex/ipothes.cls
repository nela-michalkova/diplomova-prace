%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This is file `ipothes.cls' (an acronym for Institute of      %%
%% Physics in Opava THESis), a LaTeX document class file for    %%
%% preparing Final (Bachelor, Master, Rigorous, Doctoral,       %%
%% Habilitation, and Professor) Theses.                         %%
%%                                                              %%
%% Based on:                                                    %%
%% * physuth.cls, version 0.74 released 2009/04/24 (and ad hoc  %%  
%%   workaround for author's Habilitation Thesis, 2014) which   %%
%%   is loosely based on 1999's mwbk.cls by Marcin Woli\'nski   %%
%% * fpfslut.cls, version 0.8 released 2015/07/26               %%
%%                                                              %%
%% This package may be distributed under the terms of the LaTeX %%
%% Project Public License, as described in lppl.txt in the base %%
%% LaTeX distribution.                                          %%
%% ------------------------------------------------------------ %%
\NeedsTeXFormat{LaTeX2e}[2021/06/01]                            %%
%% Version:                                                     %%
\def\ipotname{ipothes} \ProvidesClass{\ipotname}[2022/06/03]    %%
%% Developed & maintained by <stanislav.hledik@physics.slu.cz>  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Conditional counters
\ifx\pdfoutput\undefined \csname newcount\endcsname\pdfoutput\fi
\ifx\l@nohyphenation\@undefined \newlanguage\l@nohyphenation \fi

%% Options
\newif\if@draftsw           \@draftswfalse
\newif\if@narrowmargin      \@narrowmarginfalse
\newif\if@htext             \@htextfalse
\newif\if@fllastline        \@fllastlinefalse
\newif\if@indentfirst       \@indentfirstfalse
\newif\if@headfootsans      \@headfootsansfalse
\newif\if@mainmatter        \@mainmattertrue % used at many places
\newif\if@restonecol        % used in the `theindex' environment
\newif\ifHeadingNumbered    % used in sectioning
\DeclareOption{draft}       {\@draftswtrue%
                            \setlength{\overfullrule}{5\p@}
                            \def\@note#1{\marginpar{\raggedright\scriptsize\sffamily #1}}
                            }
\DeclareOption{final}       {\setlength{\overfullrule}{\z@}%
                            \let\@note\@gobble%
                            }
\DeclareOption{narrowmargin}{\@narrowmargintrue}
\DeclareOption{htext}       {\@htexttrue}
\DeclareOption{fllastline}  {\@fllastlinetrue}
\DeclareOption{indentfirst} {\@afterindenttrue} \@afterindentfalse
\DeclareOption{headfootsans}{\@headfootsanstrue}
\ExecuteOptions{final}
\ProcessOptions

%% Useful font setting macros
\def\sectsetup{\sffamily\bfseries}
\def\titlesetup{\sffamily}
\if@headfootsans
  \def\headsetup{\sffamily}\def\running@text{\small\slshape}
  \def\footsetup{\sffamily}
\else
  \let\headsetup\@empty\def\running@text{\small\slshape}
  \let\footsetup\@empty
\fi

%% Improved \note command (see above `draft' and `final' options)
\def\@@note#1{\marginpar{\raggedright\scriptsize\sffamily #1}}
\def\note{\@ifstar\@@note\@note}

%% Font size settings
\renewcommand*{\normalsize}{% 12/16
  \@setfontsize\normalsize\@xiipt{16}
  \abovedisplayskip 14.7\p@ \@plus3.75\p@ \@minus3.75\p@
  \abovedisplayshortskip \z@ \@plus3.75\p@
  \belowdisplayshortskip 8.5\p@ \@plus3.75\p@ \@minus3.75\p@
  \belowdisplayskip \abovedisplayskip
}
\normalsize
\newcommand*{\tiny}{\@setfontsize\tiny\@vipt\@viipt} % 6/7
\newcommand*{\scriptsize}{\@setfontsize\scriptsize\@viiipt{9.5}} % 8/9.5
\newcommand*{\footnotesize}{% 10/12
  \@setfontsize\footnotesize\@xpt\@xiipt
  \abovedisplayskip 10.8\p@ \@plus2.5\p@ \@minus2.5\p@
  \abovedisplayshortskip \z@ \@plus1.5\p@
  \belowdisplayshortskip 5\p@ \@plus2.5\p@ \@minus2.5\p@
  \belowdisplayskip \abovedisplayskip
}
\newcommand*{\small}{% 11/14
  \@setfontsize\small\@xipt{14}%
  \abovedisplayskip 12.8\p@ \@plus2.5\p@ \@minus3.5\p@
  \abovedisplayshortskip \z@ \@plus2.5\p@
  \belowdisplayshortskip 7\p@ \@plus3.5\p@ \@minus3.5\p@
  \belowdisplayskip \abovedisplayskip
}
\let\secondarysize\small
%% normalsize 12/16, see above
\newcommand*{\large}{\@setfontsize\large\@xivpt{18}} % 14/18
\newcommand*{\Large}{\@setfontsize\Large\@xviipt{22}} % 17/22
\newcommand*{\LARGE}{\@setfontsize\LARGE\@xxpt{25}} % 20/25
\newcommand*{\huge}{\@setfontsize\huge\@xxvpt{30}} % 25/30
\let\Huge=\huge

%% Dimensions and skips
\setlength{\smallskipamount}{.25\baselineskip \@plus.125\baselineskip
  \@minus.0625\baselineskip}
\setlength{\medskipamount}{.5\baselineskip \@plus.25\baselineskip
  \@minus.125\baselineskip}
\setlength{\bigskipamount}{1\baselineskip \@plus.5\baselineskip
  \@minus.25\baselineskip}
\setlength{\paperheight}     {297mm} % A4 height
\setlength{\paperwidth}      {210mm} % A4 width
\setlength{\columnsep}       {1.5pc}
\setlength{\columnseprule}   {\z@}
\if@narrowmargin
  \setlength{\@tempdima}     {23.98mm} % page margin (mirror 41 lines x 0.7071067812)
\else
  \setlength{\@tempdima}     {37.66mm} % page margin (mirror 39 lines x 0.6180339887)
\fi
  \setlength{\@tempdimb}     {\paperwidth}
  \addtolength{\@tempdimb}   {-2\@tempdima}
  \addtolength{\@tempdima}   {-1in}
\setlength{\oddsidemargin}   {\@tempdima}
\setlength{\evensidemargin}  {\@tempdima}
\setlength{\textwidth}       {\@tempdimb}
\setlength{\headheight}      {\baselineskip}
\setlength{\headsep}         {1.8\baselineskip}
\addtolength{\headsep}       {2\p@}
\setlength{\topskip}         {12\p@}
\setlength{\footskip}        {2\baselineskip}
\setlength{\maxdepth}        {.5\topskip}
\if@narrowmargin
  \setlength{\textheight}    {40\baselineskip}
  \setlength{\topmargin}     {-24\p@}
\else
  \setlength{\textheight}    {38\baselineskip}
  \setlength{\topmargin}     {-12\p@}
\fi
\addtolength{\textheight}    {\topskip}
\setlength{\footnotesep}     {12\p@}
\setlength{\skip\footins}    {12\p@ \@plus 18\p@ \@minus 2\p@}
\setlength{\floatsep}        {12\p@ \@plus 4\p@ \@minus 4\p@}
\setlength{\textfloatsep}    {20\p@ \@plus 6\p@ \@minus 4\p@}
\setlength{\intextsep}       {14\p@ \@plus 4\p@ \@minus 4\p@}
\setlength{\dblfloatsep}     {12\p@ \@plus 4\p@ \@minus 4\p@}
\setlength{\dbltextfloatsep} {20\p@ \@plus 6\p@ \@minus 4\p@}
\setlength{\@fptop}          {\z@ \@plus 1fil}
\setlength{\@fpsep}          {10\p@ \@plus 2fil}
\setlength{\@fpbot}          {\z@ \@plus 1fil}
\setlength{\@dblfptop}       {\z@ \@plus 1fil}
\setlength{\@dblfpsep}       {10\p@ \@plus 2fil}
\setlength{\@dblfpbot}       {\z@ \@plus 1fil}
\if@narrowmargin
  \setlength{\marginparsep}  {9pt} % approx. center between text and paper edge
  \setlength{\marginparwidth}{50pt}
\else
  \setlength{\marginparsep}  {1pc}
  \setlength{\marginparwidth}{5pc}
\fi
\setlength{\marginparpush}   {.5\baselineskip}

%% Page positioning macros
\def\spineoffset#1{%
  \addtolength{\oddsidemargin}{-#1}
  \addtolength{\evensidemargin}{#1}}
\def\verticalshift#1{\addtolength{\topmargin}{-#1}}

\def\markboth#1#2{{%
  \let\protect\@unexpandable@protect
  \let\label\relax \let\index\relax \let\glossary\relax
  \xdef\@themark{{#1}{#2}}%
  \mark{\@themark}}\if@nobreak\ifvmode\nobreak\fi\fi}
\def\@markright#1#2#3{\@temptokena {#1}%
  \unrestored@protected@xdef\@themark{{\the\@temptokena}{#3}}}
\def\secondarysize{\small}
\setlength{\normallineskip}{2.5\p@}
\setlength{\normallineskiplimit}{-1\p@}
\normalbaselines
\renewcommand*{\baselinestretch}{}
\setlength{\parskip}{\z@}
\if@twocolumn
  \setlength{\parindent}{1em}
\else
  \setlength{\parindent}{1.5em}
\fi
\newdimen\@parindent
\@parindent\parindent
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\hyphenpenalty=100
\exhyphenpenalty=10000
\widowpenalty 8000
\clubpenalty  5000
\displaywidowpenalty 5000

\setcounter{topnumber}                {2}
\renewcommand*{\topfraction}          {.99}
\setcounter{bottomnumber}             {2}
\renewcommand*{\bottomfraction}       {.99}
\setcounter{totalnumber}              {4}
\renewcommand*{\textfraction}         {0.0}
\renewcommand*{\floatpagefraction}    {.6}
\setcounter{dbltopnumber}             {2}
\renewcommand*{\dbltopfraction}       {.99}
\renewcommand*{\dblfloatpagefraction} {.6}
\newcommand*{\hf@empty}{%
  \let\@oddhead\@empty
  \let\@oddfoot\@empty
  \let\@evenhead\@empty
  \let\@evenfoot\@oddfoot}
\newcommand*{\hf@plain}{%
  \let\@oddhead\@empty
  \def\@oddfoot{\reset@font\footsetup\hfil\thepage\hfil}%
  \let\@evenhead\@empty
  \let\@evenfoot\@oddfoot}
\newcommand*{\hf@outer}{%
  \let\@oddhead\@empty
  \def\@oddfoot{\@empty\reset@font\footsetup\hfil\thepage}%
  \let\@evenhead\@empty
  \def\@evenfoot{{\reset@font\footsetup\thepage}\hfil\@empty}}
\newcommand*{\hf@headings}{%
  \def\@oddhead{%
    \reset@font\headsetup{\running@text\rightmark}\hfil\thepage}%
  \let\@oddfoot\@empty
  \def\@evenhead{%
    \reset@font\headsetup\thepage\hfil{\running@text\leftmark}}%
  \let\@evenfoot\@oddfoot
}
\newcommand*{\hf@uheadings}{%
  \def\@oddhead{\vtop{%
    \hb@xt@\textwidth{%
      \reset@font\headsetup{\running@text\rightmark}\hfil\thepage\strut}%
      \vskip2\p@\hrule\vskip-2.4\p@}}%
  \let\@oddfoot\@empty
  \def\@evenhead{\vtop{%
    \hb@xt@\textwidth{%
      \reset@font\headsetup\strut\thepage\hfil{\running@text\leftmark}}%
      \vskip2\p@\hrule\vskip-2.4\p@}}%
  \let\@evenfoot\@oddfoot
}
\def\ps@empty{%
  \let\@mkboth\@gobbletwo
  \let\ps@normal\hf@empty
  \let\ps@opening\hf@empty
  \let\ps@closing\hf@empty
  \let\ps@blank\hf@empty
  \ps@normal}
\def\ps@plain{%
  \let\@mkboth\@gobbletwo
  \let\ps@normal\hf@plain
  \let\ps@opening\hf@plain
  \let\ps@closing\hf@empty
  \let\ps@blank\hf@empty
  \ps@normal}
\def\ps@outer{%
  \let\@mkboth\@gobbletwo
  \let\ps@normal\hf@outer
  \let\ps@opening\hf@outer
  \let\ps@closing\hf@empty
  \let\ps@blank\hf@empty
  \ps@normal}
  \def\ps@headings{%
    \let\@mkboth\markboth
    \let\ps@normal\hf@headings
    \let\ps@opening\hf@empty
    \let\ps@closing\hf@headings
    \let\ps@blank\hf@empty
    \ps@normal
    \def\chaptermark##1{%
      \markboth{%
        \ifHeadingNumbered
          \if@mainmatter
            \@chapapp\
              \thechapter.\enspace
          \fi
        \fi
      ##1}{}}%
    \def\sectionmark##1{%
      \markright{%
        \ifHeadingNumbered \thesection.\enspace \fi
        ##1}}}
\def\ps@myheadings{%
  \let\ps@normal\hf@headings
  \let\ps@opening\hf@empty
  \let\ps@closing\hf@headings
  \let\ps@blank\hf@empty
  \ps@normal
  \let\@mkboth\@gobbletwo
  \let\chaptermark\@gobble
  \let\sectionmark\@gobble
}
\def\ps@uheadings{%
  \let\@mkboth\markboth
  \let\ps@normal\hf@uheadings
  \let\ps@opening\hf@plain
  \let\ps@closing\hf@uheadings
  \let\ps@blank\hf@empty
  \ps@normal
  \def\chaptermark##1{%
    \markboth{%
      \ifHeadingNumbered
        \if@mainmatter
          \@chapapp\
            \thechapter.\enspace
        \fi
      \fi
      ##1}{}}%
  \def\sectionmark##1{%
    \markright{%
      \ifHeadingNumbered \thesection.\enspace \fi
      ##1}}%
}
\def\ps@myuheadings{%
  \let\ps@normal\hf@uheadings
  \let\ps@opening\hf@empty
  \let\ps@closing\hf@uheadings
  \let\ps@blank\hf@empty
  \ps@normal
  \let\@mkboth\@gobbletwo
  \let\chaptermark\@gobble
  \let\sectionmark\@gobble
}
\let\@date\@empty
\let\@frontispice\null
\newcommand*{\maketitle}{%
  \frontmatter
  \bgroup
    \setcounter{page}{\@ne}
    \titlesetup
    \begin{center}   
      \thispagestyle{empty}
      \vspace*{19.5mm}\par
      {\Large\bfseries\@title\par}
      \ifx\@undefined\@subtitle\relax\else%
        {\vspace{3ex}\par\large\bfseries\@subtitle\par}\fi
      \vspace{5ex}\par
      {\large\bfseries\@author\par}
      \vspace{8ex}\par
      {\bfseries\MakeUppercase{\@thesiskind}}
    \end{center}
    \clearpage
    \@frontispice
    \thispagestyle{empty}\null
    \newpage
    \begin{center}    
      \thispagestyle{empty}
      {\bfseries\MakeUppercase{\@institution}\par}
      \vspace{3.5ex}\par
      \includegraphics[width=44.893333mm]{\@emblem}% 0.333333*narrowtext
      \par\vspace{5.0ex}\par 
      {\huge\bfseries\@title\par}
      \ifx\@undefined\@subtitle\relax\else%
        {\vspace{3.5ex}\par\Large\bfseries\@subtitle\par}\fi
      \vspace{8ex}\par
      {\Large\bfseries\@author\par}
      \vfill\par
      \ifx\@supervisor\@empty\relax\else%
        {\large\bfseries\@supervisorname: \@supervisor\par}\fi
      \ifx\@undefined\@advisor\relax\else%
        {\vspace{6\p@}\par\large\bfseries\@advisorname: \@advisor\par}\fi
      \ifx\@undefined\@referee\relax\else%
        {\vspace{6\p@}\par\large\bfseries\@refereename: \@referee\par}\fi
      \ifx\@undefined\@program\relax\else%
        {\vspace{30\p@}\par\bfseries\@programname: \@program\par}\fi
      \ifx\@undefined\@specialization\relax\else%
        {\ifx\@undefined\@program\vspace{18\p@}\fi%
        \par\vspace{6\p@}\bfseries\@specializationname: \@specialization\par}\fi
      \vfill\par\null\par
      {\bfseries\hspace*{1.5em}\MakeUppercase{\@thesiskind}\hfill%
        \MakeUppercase{\@seat}\ \@yearofsubmission\hspace*{1.5em}\par}
    \end{center}
  \egroup
  \setcounter{footnote}{0}%
  \global\let\author\relax
  \global\let\@title\@empty \global\let\title\relax
  \global\let\@subtitle\@empty \global\let\subtitle\relax
  \global\let\@thesiskind\@empty \global\let\thesiskind\relax
  \global\let\@frontispice\@empty
  \global\let\frontispice\relax \global\let\frontispiceimage\relax
  \global\let\@institution\@empty \global\let\institution\relax
  \global\let\@emblem\@empty \global\let\emblem\relax
  \global\let\@program\@empty \global\let\program\relax
  \global\let\@specialization\@empty \global\let\specialization\relax
  \global\let\@supervisor\@empty \global\let\supervisor\relax
  \global\let\@advisor\@empty \global\let\advisor\relax
  \global\let\@referee\@empty \global\let\referee\relax
  \global\let\@seat\@empty \global\let\seat\relax
  \global\let\@yearofsubmission\@empty \global\let\yearofsubmission\relax
  \global\let\@pdfinfotitle\@empty \global\let\pdfinfotitle\relax
  \global\let\@pdfinfocreator\@empty \global\let\pdfinfocreator\relax
  \global\let\@pdfinfoproducer\@empty \global\let\pdfinfoproducer\relax
  \global\let\@pdfinfoauthor\@empty \global\let\pdfinfoauthor\relax
  \global\let\@pdfinfocreationdate\@empty \global\let\pdfinfocreationdate\relax
  \global\let\@pdfinfomoddate\@empty \global\let\pdfinfomoddate\relax
  \global\let\@pdfinfosubject\@empty \global\let\pdfinfosubject\relax
  \global\let\@pdfinfokeywords\@empty \global\let\pdfinfokeywords\relax
  \global\let\@thanks\@empty \global\let\thanks\relax
  \global\let\@date\@empty \global\let\date\relax
  \global\let\maketitle\relax
}

\def\@sect#1#2#3#4#5#6[#7]#8{%
  \ifnum #2>\c@secnumdepth
    \let\@svsec\@empty
  \else
    \refstepcounter{#1}%
    \protected@edef\@svsec{\@seccntformat{#1}\relax}%
  \fi
  \@tempskipa #5\relax
  \ifdim \@tempskipa>\z@
    \begingroup
      \normalfont\sectsetup#6{%
        \language\l@nohyphenation \raggedright
        \interlinepenalty \@M
        \leftskip #3\noindent
        \@svsec\ignorespaces
        #8\@@par}%
    \endgroup
    \csname #1mark\endcsname{#7}%
    \addcontentsline{toc}{#1}{%
      \ifnum #2>\c@secnumdepth \else
        \protect\numberline{\csname the#1\endcsname.}%
      \fi
      #8}%
  \else
    \def\@svsechd{%
      \normalfont\sectsetup#6{\hskip #3\relax
      \@svsec\ignorespaces #8}%
      \csname #1mark\endcsname{#7}%
      \addcontentsline{toc}{#1}{%
        \ifnum #2>\c@secnumdepth \else
          \protect\numberline{\csname the#1\endcsname.}%
        \fi
        #8}}%
  \fi
  \@xsect{#5}}
\def\@ssect#1#2#3#4#5{%
  \@tempskipa #3\relax
  \ifdim \@tempskipa>\z@
    \begingroup
      \normalfont\sectsetup#4{%
        \language\l@nohyphenation \raggedright
        \interlinepenalty \@M
        \leftskip #1\noindent\ignorespaces
        #5\@@par}%
    \endgroup
  \else
    \def\@svsechd{\normalfont\sectsetup#4{\hskip #1\relax #5}}%
  \fi
  \@xsect{#3}}
\newcommand*{\partmark}[1]{\markboth{}{}}
\newcommand*{\chaptermark}[1]{}
\newcounter{part}
\newcounter{chapter}
\newcounter{section}[chapter]
\newcounter{subsection}[section]
\newcounter{subsubsection}[subsection]
\newcounter{paragraph}[subsubsection]
\newcounter{subparagraph}[paragraph]
\renewcommand*{\thepart}         {\@Roman\c@part}
\renewcommand*{\thechapter}      {\@arabic\c@chapter}
\renewcommand*{\thesection}      {\thechapter.\@arabic\c@section}
\renewcommand*{\thesubsection}   {\thesection.\@arabic\c@subsection}
\renewcommand*{\thesubsubsection}{\thesubsection .\@arabic\c@subsubsection}
\renewcommand*{\theparagraph}    {\thesubsubsection.\@arabic\c@paragraph}
\renewcommand*{\thesubparagraph} {\theparagraph.\@arabic\c@subparagraph}
\newcommand*{\@chapapp}          {\chaptername}
\renewcommand*{\@seccntformat}[1]{\csname the#1\endcsname.\enspace}
\def\cleardoublepage{\clearpage
  \ifodd\c@page\else
    \hbox{}\thispagestyle{blank}\newpage
    \if@twocolumn\hbox{}\newpage\fi
  \fi
}
\newcommand*{\frontmatter}{\@mainmatterfalse \pagestyle{plain}
  \pagenumbering{roman}}
\newcommand*{\mainmatter}{\cleardoublepage \@mainmattertrue
  \pagestyle{uheadings} \pagenumbering{arabic}
  \setcounter{page}{\@ne}}
\newcommand*{\backmatter}{\cleardoublepage \@mainmatterfalse}
\newcommand*{\mw@getflags}[1]{%
  \expandafter\expandafter\expandafter
  \mw@getflagsx\csname#1@flags\endcsname}
\newcommand*{\mw@getflagsx}[4]{%
  \def\mw@HeadingRunIn{1#1}%
  \def\mw@HeadingBreakBefore{1#2}%
  \def\mw@HeadingBreakAfter{1#3}%
  \def\mw@HeadingWholeWidth{1#4}%
}
\newcommand*{\mw@setflags}[1]{%
  \expandafter\edef\csname#1@flags\endcsname{%
    \if\mw@HeadingRunIn 1\else 0\fi
    \if\mw@HeadingBreakBefore 1\else 0\fi
    \if\mw@HeadingBreakAfter 1\else 0\fi
    \if\mw@HeadingWholeWidth 1\else 0\fi
  }
}
\newcommand*{\mw@section}[2]{%
  \mw@getflags{#1}%
  \if@noskipsec \leavevmode \fi
  \par
  \if@nobreak \everypar{}\fi
  \@ifstar
    {\HeadingNumberedfalse\@dblarg{\mw@sectionx{#1}{#2}}}%
    {\HeadingNumberedtrue\@dblarg{\mw@sectionx{#1}{#2}}}}
\def\mw@sectionx#1#2[#3]{%
  \@dblarg{\mw@sectionxx{#1}{#2}{#3}}}
\def\mw@sectionxx#1#2#3[#4]#5{%
  \ifHeadingNumbered
    \ifnum #2>\c@secnumdepth \HeadingNumberedfalse \fi
    \if@mainmatter\else \HeadingNumberedfalse \fi
  \fi
  \ifHeadingNumbered
    \refstepcounter{#1}%
    \protected@edef\HeadingNumber{\csname the#1\endcsname\relax}%
  \else
    \let\HeadingNumber\@empty
  \fi
  \def\HeadingRHeadText{#3}%
  \def\HeadingTOCText{#4}%
  \def\HeadingText{#5}%
  \def\mw@HeadingLevel{#2 }% space delimits level number!
  \def\mw@HeadingType{#1}%
  \if\mw@HeadingBreakBefore
    \if@specialpage\else\thispagestyle{closing}\fi
    \cleardoublepage
    \if\mw@HeadingBreakAfter
      \thispagestyle{blank}\else
      \thispagestyle{opening}\fi
       \global\@topnum\z@
    \fi
  \if\mw@HeadingRunIn
    \mw@runinheading
  \else
    \if\mw@HeadingWholeWidth
      \if@twocolumn
        \if\mw@HeadingBreakAfter
        \onecolumn
        \mw@normalheading
        \pagebreak\relax
        \null
        \thispagestyle{blank}%
        \newpage
        \twocolumn
        \else
          \@topnewpage[\mw@normalheading]%
        \fi
      \else
        \mw@normalheading
      \fi
    \else
      \mw@normalheading
    \fi
  \fi
}
\newcommand*{\mw@markandtoc}{%
  \csname \mw@HeadingType mark\endcsname{\HeadingRHeadText}%
  \ifnum \mw@HeadingLevel>\c@tocdepth \else
    \expandafter\ifx\csname \mw@HeadingType @toc\endcsname\relax
      \addcontentsline{toc}{\mw@HeadingType }{\ifHeadingNumbered
        \protect\numberline{\mw@seccntformat{\HeadingNumber}}\fi
        \HeadingTOCText}%
                \else
        \csname \mw@HeadingType @toc\endcsname
                \fi
  \fi
}
\newcommand*{\mw@seccntformat}[1]{#1.\enspace}
\newcommand*{\mw@normalheading}{%
  \if\mw@HeadingBreakBefore \null \else \addpenalty\@secpenalty \fi
  \addvspace{\csname \mw@HeadingType @preskip\endcsname}%
  \bgroup\language\l@nohyphenation
    \raggedright
    \interlinepenalty\@M
    \normalfont\sectsetup
    \csname \mw@HeadingType @head\endcsname
    \@@par
  \egroup
  \mw@markandtoc
  \nobreak
  \vskip\csname\mw@HeadingType @postskip\endcsname
  \@afterheading
}
\newcommand*{\mw@runinheading}{%
  \addvspace{\csname \mw@HeadingType @preskip\endcsname}%
  \@nobreakfalse
  \global\@noskipsectrue
  \everypar{%
    \if@noskipsec
      \global\@noskipsecfalse
      {\setbox\z@\lastbox}%
      \clubpenalty\@M
      \bgroup\language\l@nohyphenation
        \normalfont\sectsetup
        \csname \mw@HeadingType @head\endcsname
      \egroup
      \mw@markandtoc
      \hspace{\csname\mw@HeadingType @postskip\endcsname}%
    \else
      \clubpenalty \@clubpenalty
      \everypar{}%
    \fi}
}
\let\@titleindent\z@

\newcommand*{\FormatBlockHeading}[1]{%
  \leftskip\@titleindent
  #1{\noindent
  \ifHeadingNumbered\mw@seccntformat\HeadingNumber\fi
  \ignorespaces\HeadingText\@@par}
}
\newcommand*{\FormatHangHeading}[1]{%
  #1{%
    \@hangfrom{\hskip\@titleindent
      \ifHeadingNumbered\mw@seccntformat\HeadingNumber\fi}%
    \ignorespaces\HeadingText\@@par}%
}
\newcommand*{\FormatChapterHeading}[3]{%
  \ifHeadingNumbered
    #2\HeadingNumber\par\nobreak
    \vspace{#1}%
  \fi
  #3\HeadingText\par\nobreak
}
\newcommand*{\FormatRunInHeading}[1]{%
  \hskip\@titleindent
  #1{%
    \ifHeadingNumbered\mw@seccntformat\HeadingNumber\fi
    \ignorespaces\HeadingText\unskip}%
}
\newcommand*{\DeclareSectioningCommand}[6][default]{%
  \def\mw@HeadingRunIn{10}\def\mw@HeadingBreakBefore{10}%
  \def\mw@HeadingBreakAfter{10}\def\mw@HeadingWholeWidth{10}%
  \mw@processflags#1,\relax
  \expandafter\def\csname #2\endcsname{\mw@section{#2}{#3}}%
  \expandafter\def\csname #2@preskip\endcsname{#4}%
  \expandafter\def\csname #2@head\endcsname{#5}%
  \expandafter\def\csname #2@postskip\endcsname{#6}%
  \mw@setflags{#2}%
}
\def\mw@processflags#1,#2\relax{%
  \expandafter\ifx\csname sec@@flag@#1\endcsname\relax
    \ClassError{\ipotname}{unknown sectioning flag `#1'}{}\else
  \csname sec@@flag@#1\endcsname\fi
  \if,#2,\else \mw@processflags#2\relax \fi
}
\def\sec@@flag@default{}
\def\sec@@flag@runin{\def\mw@HeadingRunIn{11}}
\def\sec@@flag@breakbefore{\def\mw@HeadingBreakBefore{11}}
\def\sec@@flag@breakafter{\def\mw@HeadingBreakAfter{11}}
\def\sec@@flag@wholewidth{\def\mw@HeadingWholeWidth{11}}
\DeclareSectioningCommand[breakbefore,breakafter,wholewidth]{part}{-1}
  {\z@ \@plus1fil}
  {\FormatChapterHeading{20\p@}%
  {\centering\large\partname\space}%
  {\centering\LARGE}}
  {\z@ \@plus1.9fil}
\DeclareSectioningCommand[breakbefore,wholewidth]{chapter}{0}
  {56\p@}
  {\FormatChapterHeading{12\p@}{\large\@chapapp\space}{\LARGE}}
  {24\p@ \@plus12\p@}
\newcommand*{\chapter@toc}{%
  \ifHeadingNumbered\typeout{\ipotname : \@chapapp\space\thechapter}\fi
  \addcontentsline{toc}{chapter}{%
    \ifHeadingNumbered
      \mw@seccntformat{\@chapapp\ \HeadingNumber}%
    \fi
    \HeadingTOCText}%
  \addtocontents{lof}{\protect\addvspace{10\p@}}%
  \addtocontents{lot}{\protect\addvspace{10\p@}}%
}
\DeclareSectioningCommand{section}{1}
  {29\p@\@plus6\p@\@minus6\p@}
  {\FormatHangHeading{\large}}
  {14.75\p@\@plus3.75\p@\@minus3.75\p@}
\DeclareSectioningCommand{subsection}{2}
  {14.75\p@\@plus3.75\p@\@minus3.75\p@}
  {\FormatHangHeading{}}
  {14.75\p@\@plus2.5\p@\@minus2.5\p@}
\DeclareSectioningCommand{subsubsection}{3}
  {14.75\p@\@plus2.5\p@\@minus1.25\p@}
  {\FormatHangHeading{}}
  {\z@\@plus3.0\p@}
\DeclareSectioningCommand[runin]{paragraph}{4}
  {14.75\p@\@plus4\p@\@minus2.75\p@}
  {\FormatRunInHeading{}}
  {1.0em}
\DeclareSectioningCommand[runin]{subparagraph}{5}
  {8\p@\@plus4\p@\@minus2\p@}
  {\FormatRunInHeading{\mdseries\itshape}}
  {0.75em}
\newcommand{\bibsection}[2][]{#1\item[\large\sectsetup #2] \item[]
  \addcontentsline{toc}{section}{#2}}
\setlength{\leftmargini}    {\parindent}
\setlength{\leftmargin}     {\leftmargini}
\setlength{\leftmarginii}   {\parindent}
\setlength{\leftmarginiii}  {\parindent}
\setlength{\leftmarginiv}   {\parindent}
\if@twocolumn
  \setlength{\leftmarginv}  {.5em}
  \setlength{\leftmarginvi} {.5em}
\else
  \setlength{\leftmarginv}  {1em}
  \setlength{\leftmarginvi} {1em}
\fi
\setlength{\labelsep}       {.5em}
\setlength{\labelwidth}     {\leftmargini}
\addtolength{\labelwidth}   {-\labelsep}
\setlength{\partopsep}      {\z@}
\@beginparpenalty         -\@lowpenalty
\@endparpenalty           -\@lowpenalty
\@itempenalty             -\@lowpenalty
\def\@listi{%
  \leftmargin\leftmargini
  \topsep    .5\baselineskip\@plus.25\baselineskip\@minus.125\baselineskip
  \parsep    \z@
  \itemsep   \z@}
\@listi
\def\@listii{%
  \leftmargin\leftmarginii
  \labelwidth\leftmarginii
  \advance\labelwidth-\labelsep
  \topsep    \z@
  \parsep    \z@
  \itemsep   \z@}
\def\@listiii{%
  \leftmargin\leftmarginiii
  \labelwidth\leftmarginiii
  \advance\labelwidth-\labelsep
  \itemsep   \z@}
\def\@listiv{%
  \leftmargin\leftmarginiv
  \labelwidth\leftmarginiv
  \advance\labelwidth-\labelsep}
\def\@listv{%
  \leftmargin\leftmarginv
  \labelwidth\leftmarginv
  \advance\labelwidth-\labelsep}
\def\@listvi{%
  \leftmargin\leftmarginvi
  \labelwidth\leftmarginvi
  \advance\labelwidth-\labelsep}
\def\enumerate{%
  \ifnum \@enumdepth >\thr@@\@toodeep\else
    \advance\@enumdepth\@ne
    \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
      \expandafter
      \list
        \csname label\@enumctr\endcsname
        {\partopsep\topsep \topsep\z@ %MW
        \usecounter{\@enumctr}\def\makelabel##1{##1\hfil}}%
  \fi}
\newenvironment*{enumerate*}{%
  \leftmargini\z@ % ~sh
  \@listi         % ~sh
  \ifnum \@enumdepth >\thr@@\@toodeep\else
    \advance\@enumdepth\@ne
    \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
      \expandafter
      \list
        \csname label\@enumctr\endcsname
        {\partopsep\topsep \topsep\z@ %MW
%        \leftmargin\z@ % ~sh
        \labelwidth\z@
%        \ifnum\@enumdepth >\@ne % ~sh
%          \itemindent\labelsep
%        \else
          \itemindent\@parindent \advance\itemindent\labelsep
%        \fi
        \usecounter{\@enumctr}\def\makelabel##1{##1\hfil}}%
  \fi}
  {\endlist}
\renewcommand*{\theenumi}{\@arabic\c@enumi}
\renewcommand*{\theenumii}{\@alph\c@enumii}
\renewcommand*{\theenumiii}{\@roman\c@enumiii}
\renewcommand*{\theenumiv}{\@Alph\c@enumiv}
\newcommand*{\labelenumi}{\theenumi.}
\newcommand*{\labelenumii}{(\theenumii)}
\newcommand*{\labelenumiii}{\theenumiii.}
\newcommand*{\labelenumiv}{\theenumiv.}
\renewcommand*{\p@enumii}{\theenumi}
\renewcommand*{\p@enumiii}{\theenumi\theenumii)}
\renewcommand*{\p@enumiv}{\p@enumiii\theenumiii}
\def\itemize{%
  \ifnum \@itemdepth >\thr@@\@toodeep\else
    \advance\@itemdepth\@ne
    \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
    \expandafter
    \list
      \csname\@itemitem\endcsname
      {\partopsep\topsep \topsep\z@ %MW
      \def\makelabel##1{##1\hfil}}%
  \fi}
\newenvironment*{itemize*}{%
  \leftmargini\z@ % ~sh
  \@listi         % ~sh
  \ifnum \@itemdepth >\thr@@\@toodeep\else
    \advance\@itemdepth\@ne
    \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
    \expandafter
    \list
      \csname\@itemitem\endcsname
      {\partopsep\topsep \topsep\z@ %MW
%      \leftmargin\z@ % ~sh
      \labelwidth\z@
%      \ifnum\@itemdepth >\@ne % ~sh
%        \itemindent\labelsep
%      \else
        \itemindent\@parindent \advance\itemindent\labelsep
%      \fi
      \def\makelabel##1{##1\hfil}}%
  \fi}
  {\endlist}
\newcommand*{\labelitemi}{\textbullet}
\newcommand*{\labelitemii}{\normalfont\bfseries\textendash}
\newcommand*{\labelitemiii}{\textasteriskcentered}
\newcommand*{\labelitemiv}{\textperiodcentered}
\newenvironment*{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*{\descriptionlabel}[1]{\hspace\labelsep
                                   \normalfont\bfseries #1}
\newenvironment*{description*}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\stardescriptionlabel}}
               {\endlist}
\newcommand*{\stardescriptionlabel}[1]{\hspace\labelsep
                                       \normalfont\itshape #1}

\newenvironment*{verse}{%
  \let\\\@centercr
  \list{}{\itemsep\z@
          \parsep \medskipamount
          \itemindent -1.5em%
          \listparindent\itemindent
          \rightmargin\leftmargin
          \advance\leftmargin 1.5em}%
          \item\relax}
  {\endlist}
\newenvironment*{quotation}{%
  \list{}{\listparindent 1.5em%
          \itemindent \listparindent
          \rightmargin \leftmargin
          \parsep \z@ \@plus\p@}%
          \item\relax}
  {\endlist}
\newenvironment*{quote}{%
  \list{}{\rightmargin\leftmargin}%
          \item\relax}
  {\endlist}

\newenvironment*{annotation*}[1][\bbl@main@language]{%
  \cleardoublepage\vspace*{19mm}\par
  \ifx#1\bbl@main@language\relax\else\selectlanguage{#1} \input{ipotbabel}\fi
  \begin{center}\titlesetup\bfseries\@annotationname\end{center}\par
  \list{}{\setlength{\listparindent}{\parindent}
          \setlength{\leftmargin}{\z@}
          \setlength{\rightmargin}{\z@}
          \thispagestyle{empty}
  \itshape}\item\relax\centeredlast}
  {\endlist}
\newenvironment*{annotation}[1][\bbl@main@language]{%
  \vspace*{.5\baselineskip}\par
  \ifx#1\bbl@main@language\relax\else\selectlanguage{#1} \input{ipotbabel}\fi
  \begin{center}\titlesetup\bfseries\@annotationname\end{center}\par
  \list{}{\setlength{\listparindent}{\parindent}
          \setlength{\leftmargin}{\z@}
          \setlength{\rightmargin}{\z@}
  \itshape}\item\relax\centeredlast}
  {\endlist}

\newenvironment*{keywords}[1][\bbl@main@language]{%
  \vspace*{.5\baselineskip}\par
  \ifx#1\bbl@main@language\relax\else\selectlanguage{#1} \input{ipotbabel}\fi
  \begin{center}\titlesetup\bfseries\@keywordsname\end{center}\par
  \list{}{\setlength{\listparindent}{\parindent}
          \setlength{\leftmargin}{\z@}
          \setlength{\rightmargin}{\z@}
  \itshape}\item\relax\centeredlast\hyphenpenalty=1000}
  {\endlist}

\newenvironment*{acknowledgement}[1][\bbl@main@language]{%
  \cleardoublepage\vspace*{19mm}\par
  \ifx#1\bbl@main@language\relax\else\selectlanguage{#1} \input{ipotbabel}\fi
  \begin{center}\titlesetup\bfseries\@acknowledgementname\end{center}\par
  \list{}{\setlength{\listparindent}{\parindent}
          \setlength{\leftmargin}{\z@}
          \setlength{\rightmargin}{\z@}
          \thispagestyle{empty}
  \itshape}\item\relax\centeredlast}
  {\endlist}

\newenvironment{declaration}[2][\bbl@main@language]{%
  \cleardoublepage\vspace*{19mm}\par
  \ifx#1\bbl@main@language\relax\else\selectlanguage{#1} \input{ipotbabel}\fi
  \begin{center}\titlesetup\bfseries\@declarationname\end{center}\par
  \list{}{\setlength{\listparindent}{\parindent}
          \setlength{\leftmargin}{\z@}
          \setlength{\rightmargin}{\z@}
          \thispagestyle{empty}
  }\item\relax\centeredlast\def\@declared@in{#2}}
  {\par\vfill\par\noindent
    \@declared@in\hfill\makebox[.6\linewidth][c]{\dotfill}\par\vskip-1.75ex\par
    \null\hfill\makebox[.6\linewidth][c]{\@author}\par
    \endlist%
    \global\let\@author\@empty \global\let\@declared@in\@empty}

\newenvironment*{dedication}[1][\bbl@main@language]{%
  \cleardoublepage\vspace*{26.5mm}\par
  \ifx#1\bbl@main@language\relax\else\selectlanguage{#1}\fi
  \list{}{\setlength{\listparindent}{\parindent}
          \setlength{\leftmargin}{\z@}
          \setlength{\rightmargin}{\z@}
          \thispagestyle{empty}
  \itshape}\item\relax\centering}
  {\endlist}
\newenvironment*{dedication*}[1][\bbl@main@language]{%
  \cleardoublepage\vspace*{19mm}\par
  \ifx#1\bbl@main@language\relax\else\selectlanguage{#1} \input{ipotbabel}\fi
  \begin{center}\titlesetup\bfseries\@dedicationname\end{center}\par
  \list{}{\setlength{\listparindent}{\parindent}
          \setlength{\leftmargin}{\z@}
          \setlength{\rightmargin}{\z@}
          \thispagestyle{empty}
  \itshape}\item\relax\centering}
  {\endlist}

\newenvironment{motto}[2][\bbl@main@language]{%
  \ifx#1\bbl@main@language\relax\else\selectlanguage{#1}\fi
  \list{}{\setlength{\listparindent}{\z@}
          \setlength{\leftmargin}{\z@}
          \setlength{\rightmargin}{.333\textwidth}
  \raggedright\small\itshape}\item\relax
  \def\@motto@author{#2}}
  {\par\vspace{1ex}\par
    \hskip\@parindent\textemdash\thinspace\@motto@author\endlist%
    \global\let\@motto@author\@empty}

\newcommand*{\appendix}{\par
  \setcounter{chapter}{0}\setcounter{section}{0}%
  \input{ipotbabel}%
  \gdef\@chapapp{\appendixname}%
  \gdef\thechapter{\@Alph\c@chapter}}
\setlength{\arraycolsep}{5\p@}
\setlength{\tabcolsep}{6\p@}
\setlength{\arrayrulewidth}{.4\p@}
\setlength{\doublerulesep}{2\p@}
\setlength{\tabbingsep}{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength{\fboxsep}{3\p@}
\setlength{\fboxrule}{.4\p@}
\@addtoreset {equation}{chapter}
\renewcommand*{\theequation}%
  {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@equation}
\def\@floatboxreset{%
  \reset@font
  \normalsize
  \@setminipage
  \csname\@captype setup\endcsname
}
\newcounter{figure}[chapter]
\renewcommand*{\thefigure}%
  {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename~\thefigure}
\newenvironment{figure}
  {\@float{figure}}
  {\end@float}
\newenvironment{figure*}
  {\@dblfloat{figure}}
  {\end@dblfloat}
\newcounter{table}[chapter]
\renewcommand*{\thetable}%
  {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename~\thetable}
\newenvironment{table}
  {\@float{table}}
  {\end@float}
\newenvironment{table*}
  {\@dblfloat{table}}
  {\end@dblfloat}
\if@fllastline
  \let\centeredlast\relax
\else
  \def\centeredlast{%
    \let\\\@centercr
    \rightskip\z@\@plus-.5fil
    \leftskip\z@\@plus.5fil
    \parindent\z@
    \parfillskip\@flushglue
  }
\fi
\newlength{\abovecaptionskip}
\newlength{\belowcaptionskip}
\setlength{\abovecaptionskip}{.5\baselineskip}
\setlength{\belowcaptionskip}{.5\baselineskip}
\long\def\@caption#1[#2]#3{%
  \par
  \addcontentsline{\csname ext@#1\endcsname}{#1}%
    {\protect\numberline{\csname the#1\endcsname.}{\ignorespaces #2}}%
  \begingroup
    \@parboxrestore
    \if@minipage
      \@setminipage
    \fi
    \@makecaption{\csname fnum@#1\endcsname}{\ignorespaces #3}\par
  \endgroup}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*{\cal}{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*{\mit}{\@fontswitch\relax\mathnormal}
\newcommand*{\@pnumwidth}{1.667em}
\newcommand*{\@tocrmarg}{2.667em \@plus 2cm}
\newcommand*{\@dotsep}{4.5}
\newcommand*{\tableofcontents}{%
  {\@tempcnta=\c@tocdepth
  \c@tocdepth=-1
  \chapter*{\contentsname}%
  \c@tocdepth=\@tempcnta
  \secondarysize
  \gdef\previous@toc@level{-1000}%
  \@starttoc{toc}}%
}
\def\mw@tocskip#1#2{%
  \ifnum#1>\previous@toc@level
    \nobreak
  \else\ifnum#1<\previous@toc@level
    \addpenalty\@secpenalty
  \fi\fi
  \gdef\previous@toc@level{#1}%
  \vskip #2\relax
}
\def\mw@tocline#1#2#3#4#5{%
  \ifnum #1>\c@tocdepth \else
    \mw@tocskip{#1}{\z@ \@plus.2\p@}%
    {\leftskip #2\relax
    \@tempdima #3\relax
    \advance\leftskip \@tempdima
    \rightskip \@tocrmarg \parfillskip -\rightskip
    \interlinepenalty\@M
    \noindent\kern-\@tempdima{\ignorespaces#4}\nobreak
    \leaders\hbox{$\m@th
      \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
      mu$}\hfill
    \nobreak
    \hb@xt@\@pnumwidth{\hfil\normalfont \normalcolor #5}%
    \par}%
  \fi}
\def\numberline#1{\hb@xt@\@tempdima{#1\hfil}\ignorespaces}
\newcommand*{\l@part}[2]{% SOME MORE HACKS
  \ifnum \c@tocdepth >-2\relax
%    \mw@tocskip{-1}{.6\baselineskip}
    \mw@tocskip{-1}{\baselineskip}
    \begingroup
      \parindent\z@ \rightskip\@pnumwidth
      \parfillskip -\@pnumwidth
%      {\leavevmode\normalsize
      {\leavevmode
%        \def\numberline##1{##1\enspace}%
        \def\numberline##1{\raggedright\partname\ ##1}%
%        \hfil\bfseries #1\hfil\null %\hb@xt@\@pnumwidth{\hss #2}%
        \bfseries #1\hfill\null%
      }\par
      \nobreak
      \global\@nobreaktrue
      \everypar{\global\@nobreakfalse\everypar{}}%
      \endgroup
%      \addvspace{.4\baselineskip}%
      \addvspace{\baselineskip}%
  \fi}
\newcommand*{\l@chapter}[2]{\addvspace{.3\baselineskip}%
  \mw@tocline{0}{\z@}{1.5em}{\textbf{#1}}{#2}%
  \addvspace{.2\baselineskip}}
\newcommand*{\l@section}{\mw@tocline{1}{1.5em}{2.5em}}
\newcommand*{\l@subsection}{\mw@tocline{2}{4.0em}{3.25em}}
\newcommand*{\l@subsubsection}{\mw@tocline{3}{7.25em}{4.0em}}
\newcommand*{\l@paragraph}{\mw@tocline{4}{11.25em}{4.75em}}
\newcommand*{\l@subparagraph}{\mw@tocline{5}{16.0em}{5.5em}}
\newcommand*{\listoffigures}{%
  \chapter*{\listfigurename}
  {\secondarysize
  \gdef\previous@toc@level{-1000}%
  \@starttoc{lof}}%
}
\newcommand*{\l@figure}{\mw@tocline{1}{\z@}{3.5em}}
\newcommand*{\listoftables}{%
  \chapter*{\listtablename}
  {\secondarysize
  \gdef\previous@toc@level{-1000}%
  \@starttoc{lot}}%
}
\let\l@table\l@figure

\newenvironment*{thebibliography}[1]
  {\section*{\bibname}\list{\@biblabel{\@arabic\c@enumiv}}{}}{\endlist}

%%%% load etoolbox package  %%%%
\RequirePackage{etoolbox} %[2020/10/05]

%%%% load natbib package %%%%
\RequirePackage[numbers,sort&compress]{natbib} %[2010/09/13]
\bibpunct[]{[}{]}{,}{n}{}{}
\renewenvironment*{thebibliography}[1]{%
  \def\chaptermark##1{\@mkboth{##1}{##1}}%
  \chapter*{\bibname}%
  \list{\@biblabel{\@arabic\c@enumiv}}%
  {%
    \settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth
    \advance\labelsep by.2em
    \advance\leftmargin\labelsep
    \setlength{\itemsep}{1ex}%
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \renewcommand*{\theenumiv}{\@arabic\c@enumiv}%
  }%
  \sloppy\raggedbottom\clubpenalty4000\widowpenalty4000%
  \sfcode`\.\@m\hbadness=2500}
  {\def\@noitemerr
    {\ClassWarning{\ipotname}{empty `thebibliography' environment}}%
  \endlist}
\newcommand*{\newblock}{\hskip.11em\@plus.33em\@minus.07em}

\newenvironment*{theindex}%
  {\if@twocolumn
    \@restonecolfalse
  \else
    \@restonecoltrue
  \fi
  \columnseprule \z@
  \columnsep 35\p@
  \twocolumn
  \chapter*{\indexname}%
  \thispagestyle{opening}\parindent\z@
  \parskip\z@ \@plus .3\p@\relax
  \let\item\@idxitem
  \secondarysize}
  {\if@restonecol\onecolumn\else
    \clearpage
  \fi}
\newcommand*{\@idxitem}{\par\hangindent 40\p@}
\newcommand*{\subitem}{\@idxitem \hspace*{20\p@}}
\newcommand*{\subsubitem}{\@idxitem \hspace*{30\p@}}
\newcommand*{\indexspace}{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand*{\footnoterule}{%
  \kern-3\p@
  \hrule\@width6pc
  \kern2.6\p@}
\@addtoreset{footnote}{chapter}
\newcommand{\@makefntext}[1]{\parindent\@parindent%
  \noindent\hb@xt@\@parindent{\hss\@makefnmark\thinspace}#1}

\def\@title{\texttt{<No \textbackslash title given>}%
  \ClassWarningNoLine{\ipotname}{no \noexpand\title given}}
\def\subtitle#1{\gdef\@subtitle{#1}}
\def\@author{\texttt{<No \textbackslash author given>}%
  \ClassWarningNoLine{\ipotname}{no \noexpand\author given}}
\def\thesiskind#1{\gdef\@thesiskind{#1}}
\long\def\frontispice#1{\gdef\@frontispice{#1}}
\newcommand{\frontispiceimage}[3][18mm]{%
  \gdef\@frontispice{%
    \begin{center}
      \vspace*{#1} \includegraphics[width=#2]{#3}
    \end{center}}%
  }
\def\institution#1{\gdef\@institution{#1}%
  \ClassWarningNoLine{\ipotname}{\noexpand\institution altered}}
\def\emblem#1{\gdef\@emblem{#1}%
  \ClassWarningNoLine{\ipotname}{\noexpand\emblem altered}}
\def\program#1{\gdef\@program{#1}}
\def\specialization#1{\gdef\@specialization{#1}}
\def\supervisor#1{\gdef\@supervisor{#1}}
  \def\@supervisor{\texttt{<No \textbackslash supervisor given>}%
    \ClassWarningNoLine{\ipotname}{no \noexpand\supervisor given}}
\def\advisor#1{\gdef\@advisor{#1}}
\def\referee#1{\gdef\@referee{#1}}
\def\seat#1{\gdef\@seat{#1}%
  \ClassWarningNoLine{\ipotname}{\noexpand\seat altered}}
\def\yearofsubmission#1{\gdef\@yearofsubmission{#1}}
\def\@yearofsubmission{\texttt{<No \textbackslash yearofsubmission>}%
  \ClassWarningNoLine{\ipotname}{no \noexpand\yearofsubmission given}}
\def\pdfinfotitle#1{\gdef\@pdfinfotitle{#1}} \let\@pdfinfotitle\@empty
\def\pdfinfocreator#1{\gdef\@pdfinfocreator{#1}} \def\@pdfinfocreator{TeX}
  \def\pdfinfoproducer#1{\gdef\@pdfinfoproducer{#1}}
\def\@pdfinfoproducer{pdfLaTeX with the \ipotname{} class}
\def\pdfinfoauthor#1{\gdef\@pdfinfoauthor{#1}} \let\@pdfinfoauthor\@empty
\def\pdfinfocreationdate#1{\gdef\@pdfinfocreationdate{#1}}
  \let\@pdfinfocreationdate\@empty
\def\pdfinfomoddate#1{\gdef\@pdfinfomoddate{#1}} \let\@pdfinfomoddate\@empty
\def\pdfinfosubject#1{\gdef\@pdfinfosubject{#1}} \let\@pdfinfosubject\@empty
\def\pdfinfokeywords#1{\gdef\@pdfinfokeywords{#1}} \let\@pdfinfokeywords\@empty

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% <to be executed at the very end of the preamble (etoolbox)> %%%%
\AtEndPreamble{ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ifx\bbl@main@language\undefined
  \def\bbl@main@language{nohyphenation}
  \let\selectlanguage\@gobble
\fi
\InputIfFileExists{ipotbabel}%
  {\typeout{** \ipotname : main language `\bbl@main@language' localization loaded:}}%
  {\ClassWarningNoLine{\ipotname}{`ipotbabel.tex' not found; defaults used}
    \def\appendixname{Appendix}
    \def\@annotationname{Abstract}
    \def\@keywordsname{Key words}
    \def\@acknowledgementname{Acknowledgements}
    \def\@declarationname{Declaration}
    \def\@dedicationname{Dedication}
    \def\@thesiskind{bachelor project}
    \def\@institution{Silesian University in  Opava\enspace
      $\boldsymbol\vert$\enspace Institute of Physics}
    \def\@programname{Program}
    \def\@specializationname{Specialization}
    \def\@supervisorname{Supervisor}
    \def\@advisorname{Advisor}
    \def\@refereename{Opponent}
    \def\@seat{Opava}
  }
\bibliographystyle{\ipotname}
\def\@emblem{emblemiop}
%% Moved from "Useful font setting macros" to override "subfig":
\def\captionsetup{\small}
%% Moved from below \@caption def above to override "subfig":
\long\def\@makecaption#1#2{%
  \addvspace\abovecaptionskip
  \bgroup\centeredlast
    {\captionsetup{\sectsetup #1.}\enspace#2\par}
  \egroup
  \vskip\belowcaptionskip}
%%%% Load hyperref/url package %%%%
%%%% Placing within AtBeginDocument causes loading as the very last %%%%
\if@htext % hypertext
  \RequirePackage[final]{hyperref} %[2022-02-21]
  \newcommand{\eprint}[2][]{ar\-X\-iv:%
    \discretionary{}{}{}\href{\@eprintBaseURL/#2}{\@eprint{#2}}}
  \def\eprintBaseURL#1{\gdef\@eprintBaseURL{#1}}
  \gdef\@eprintBaseURL{https://arxiv.org/abs}
  \newcommand{\doi}[2][]{doi:%
    \discretionary{}{}{}\href{\@doiBaseURL/#2}{\@doi{#2}}}
  \def\doiBaseURL#1{\gdef\@doiBaseURL{#1}}
  \gdef\@doiBaseURL{https://doi.org}
\else % no hypertext
  \RequirePackage{url} %[2013/09/16]
  \newcommand{\eprint}[2][]{ar\-X\-iv:\discretionary{}{}{}\@eprint{#2}}
  \newcommand{\doi}[2][]{doi:\discretionary{}{}{}\@doi{#2}}
  \newcommand{\href}[3][]{#3}
\fi
\newcommand*{\@eprint}{\begingroup \urlstyle{tt}\Url} % common
\newcommand*{\@doi}{\begingroup \urlstyle{rm}\Url}
\ifcase\pdfoutput % DVI
  \relax
\else %PDF
  \pdfcompresslevel=9    % compresses streams as page descriptions
  \pdfminorversion=5     % PDF-1.5
  \pdfobjcompresslevel=2 % compresses further data structures, requires PDF-1.5
\fi
\InputIfFileExists{\ipotname .cfg}%
  {\typeout{** \ipotname : user config file loaded:}}%
  {\ClassWarningNoLine{\ipotname}{`\ipotname .cfg' not found;^^Jcreate
      one or use preamble to place your stuff}}
\ifcase\pdfoutput % DVI
  \relax
\else %PDF
  \pdfinfo{
    /Title        (\@pdfinfotitle)
%    /Creator      (\@pdfinfocreator) % use default
%    /Producer     (\@pdfinfoproducer) % use default
    /Author       (\@pdfinfoauthor)
%    /CreationDate (\@pdfinfocreationdate) % {D:20220314000000+01'00'}
%    /ModDate      (\@pdfinfomoddate) % use compile time
    /Subject      (\@pdfinfosubject)
    /Keywords     (\@pdfinfokeywords)
  }
  \typeout{%
    pdfcompresslevel=\the\pdfcompresslevel,
    pdfminorversion=\the\pdfminorversion,
    pdfobjcompresslevel=\the\pdfobjcompresslevel}
\fi
\if@htext % hypertext
  \hypersetup{
    pdftitle={\@pdfinfotitle},%
%    pdfcreator={\@pdfinfocreator},% use default
%    pdfproducer={\@pdfinfoproducer},% use default
    pdfauthor={\@pdfinfoauthor},%
%    /CreationDate not implemented here
%    /ModDate not implemented here
    pdfsubject={\@pdfinfosubject},%
    pdfkeywords={\@pdfinfokeywords}%
  }
\fi
} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% </to be executed at the very end of the preamble (etoolbox)> %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Load ams* packages %%%%
\RequirePackage[fleqn]{amsmath} %[2021/10/15]
%\RequirePackage{amsthm} %[2020/05/29]
\RequirePackage{amssymb} %[2013/01/14]
  \setlength{\mathindent}{\z@}

%%%% Load graphicx package and path to images %%%%
\RequirePackage{graphicx} %[2021/09/16]
\graphicspath{{images/}%
              {\jobname /}%
              {\jobname /images/}}

\ifcase\pdfoutput % DVI
  \def\Gin@extensions{.eps,.ps}
\else % PDF
  \def\Gin@extensions{.pdf,.png,.jpg,.jpeg,.mps}
\fi

%%%% Path to chapter sources %%%%
\def\inpath#1{\def\input@path{#1}}
\def\input@path{{include/}%
                {\jobname /}%
                {\jobname /include/}}

%%%% Initialisation %%%%
\pagestyle{uheadings}
\pagenumbering{arabic}
\def\thefootnote{\arabic{footnote}}
\@twosidetrue
\@mparswitchtrue
\flushbottom
\onecolumn
\setcounter{footnote}{0}
\setcounter{tocdepth}{3}
\setcounter{secnumdepth}{3}

\endinput

%%
%% End of file `ipothes.cls'.
