#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.5),
    on Fri Nov 22 12:59:09 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
sound.init(rate=44100)
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.5'
expName = 'mem_search_recall'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '',
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
_loggingLevel = logging.getLevel('warning')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/gio/projects/SwitchCode/ses-002_task-stories/mem_search_recall.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('key_resp_story_instr') is None:
        # initialise key_resp_story_instr
        key_resp_story_instr = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_story_instr',
        )
    if deviceManager.getDevice('key_resp_story_start') is None:
        # initialise key_resp_story_start
        key_resp_story_start = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_story_start',
        )
    if deviceManager.getDevice('key_go_back') is None:
        # initialise key_go_back
        key_go_back = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_go_back',
        )
    # create speaker 'story'
    deviceManager.addDevice(
        deviceName='story',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=1.0
    )
    if deviceManager.getDevice('question_resp') is None:
        # initialise question_resp
        question_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='question_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "story_instructions" ---
    text_story_instr = visual.TextStim(win=win, name='text_story_instr',
        text='The experiment consists of 6 scans. \n\nDuring each scan, you will hear a story that lasts about 10 minutes. Your only task is to listen carefully to the story. Please stay still while you are listening.\n\nAfter each story, a blank screen will appear for around 20 seconds, and then you will be asked a question about the story you just heard. Please answer each question out loud in one or two sentences.',
        font='Arial',
        pos=(0, 0), height=0.04, wrapWidth=1.3, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_story_instr = keyboard.Keyboard(deviceName='key_resp_story_instr')
    
    # --- Initialize components for Routine "blank_200ms" ---
    
    # --- Initialize components for Routine "story_id" ---
    story_num = visual.TextStim(win=win, name='story_num',
        text='',
        font='Arial',
        pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_story_start = keyboard.Keyboard(deviceName='key_resp_story_start')
    
    # --- Initialize components for Routine "blank_200ms" ---
    
    # --- Initialize components for Routine "wait" ---
    text_wait = visual.TextStim(win=win, name='text_wait',
        text='wait',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_go_back = keyboard.Keyboard(deviceName='key_go_back')
    
    # --- Initialize components for Routine "stories" ---
    story = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        sampleRate=44100, 
        speaker='story',    name='story'
    )
    story.setVolume(1.0)
    text_plus = visual.TextStim(win=win, name='text_plus',
        text='+',
        font='Arial',
        pos=(0, 0), height=0.12, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "blank_15" ---
    blank_text = visual.TextStim(win=win, name='blank_text',
        text=None,
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "questions" ---
    question_text = visual.TextStim(win=win, name='question_text',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    question_resp = keyboard.Keyboard(deviceName='question_resp')
    
    # --- Initialize components for Routine "blank_200ms" ---
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "story_instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('story_instructions.started', globalClock.getTime(format='float'))
    # create starting attributes for key_resp_story_instr
    key_resp_story_instr.keys = []
    key_resp_story_instr.rt = []
    _key_resp_story_instr_allKeys = []
    # keep track of which components have finished
    story_instructionsComponents = [text_story_instr, key_resp_story_instr]
    for thisComponent in story_instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "story_instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_story_instr* updates
        
        # if text_story_instr is starting this frame...
        if text_story_instr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_story_instr.frameNStart = frameN  # exact frame index
            text_story_instr.tStart = t  # local t and not account for scr refresh
            text_story_instr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_story_instr, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_story_instr.started')
            # update status
            text_story_instr.status = STARTED
            text_story_instr.setAutoDraw(True)
        
        # if text_story_instr is active this frame...
        if text_story_instr.status == STARTED:
            # update params
            pass
        
        # *key_resp_story_instr* updates
        waitOnFlip = False
        
        # if key_resp_story_instr is starting this frame...
        if key_resp_story_instr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_story_instr.frameNStart = frameN  # exact frame index
            key_resp_story_instr.tStart = t  # local t and not account for scr refresh
            key_resp_story_instr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_story_instr, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_story_instr.started')
            # update status
            key_resp_story_instr.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_story_instr.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_story_instr.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_story_instr.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_story_instr.getKeys(keyList=['right','space','enter'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_story_instr_allKeys.extend(theseKeys)
            if len(_key_resp_story_instr_allKeys):
                key_resp_story_instr.keys = _key_resp_story_instr_allKeys[-1].name  # just the last key pressed
                key_resp_story_instr.rt = _key_resp_story_instr_allKeys[-1].rt
                key_resp_story_instr.duration = _key_resp_story_instr_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in story_instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "story_instructions" ---
    for thisComponent in story_instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('story_instructions.stopped', globalClock.getTime(format='float'))
    # check responses
    if key_resp_story_instr.keys in ['', [], None]:  # No response was made
        key_resp_story_instr.keys = None
    thisExp.addData('key_resp_story_instr.keys',key_resp_story_instr.keys)
    if key_resp_story_instr.keys != None:  # we had a response
        thisExp.addData('key_resp_story_instr.rt', key_resp_story_instr.rt)
        thisExp.addData('key_resp_story_instr.duration', key_resp_story_instr.duration)
    thisExp.nextEntry()
    # the Routine "story_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "blank_200ms" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('blank_200ms.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    blank_200msComponents = []
    for thisComponent in blank_200msComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "blank_200ms" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.2:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # is it time to end the Routine? (based on local clock)
        if tThisFlip > .2-frameTolerance:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in blank_200msComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "blank_200ms" ---
    for thisComponent in blank_200msComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('blank_200ms.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.200000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    story_loop = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('ses-002_task-stories.xlsx'),
        seed=None, name='story_loop')
    thisExp.addLoop(story_loop)  # add the loop to the experiment
    thisStory_loop = story_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisStory_loop.rgb)
    if thisStory_loop != None:
        for paramName in thisStory_loop:
            globals()[paramName] = thisStory_loop[paramName]
    
    for thisStory_loop in story_loop:
        currentLoop = story_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisStory_loop.rgb)
        if thisStory_loop != None:
            for paramName in thisStory_loop:
                globals()[paramName] = thisStory_loop[paramName]
        
        # set up handler to look after randomisation of conditions etc
        go_back = data.TrialHandler(nReps=1.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='go_back')
        thisExp.addLoop(go_back)  # add the loop to the experiment
        thisGo_back = go_back.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisGo_back.rgb)
        if thisGo_back != None:
            for paramName in thisGo_back:
                globals()[paramName] = thisGo_back[paramName]
        
        for thisGo_back in go_back:
            currentLoop = go_back
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisGo_back.rgb)
            if thisGo_back != None:
                for paramName in thisGo_back:
                    globals()[paramName] = thisGo_back[paramName]
            
            # --- Prepare to start Routine "story_id" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('story_id.started', globalClock.getTime(format='float'))
            story_num.setText('Story ' + str(story_loop.thisN + 1))
            # create starting attributes for key_resp_story_start
            key_resp_story_start.keys = []
            key_resp_story_start.rt = []
            _key_resp_story_start_allKeys = []
            # keep track of which components have finished
            story_idComponents = [story_num, key_resp_story_start]
            for thisComponent in story_idComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "story_id" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *story_num* updates
                
                # if story_num is starting this frame...
                if story_num.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    story_num.frameNStart = frameN  # exact frame index
                    story_num.tStart = t  # local t and not account for scr refresh
                    story_num.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(story_num, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'story_num.started')
                    # update status
                    story_num.status = STARTED
                    story_num.setAutoDraw(True)
                
                # if story_num is active this frame...
                if story_num.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_story_start* updates
                waitOnFlip = False
                
                # if key_resp_story_start is starting this frame...
                if key_resp_story_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_story_start.frameNStart = frameN  # exact frame index
                    key_resp_story_start.tStart = t  # local t and not account for scr refresh
                    key_resp_story_start.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_story_start, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_story_start.started')
                    # update status
                    key_resp_story_start.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_story_start.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_story_start.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_story_start.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_story_start.getKeys(keyList=['right','space','enter'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_story_start_allKeys.extend(theseKeys)
                    if len(_key_resp_story_start_allKeys):
                        key_resp_story_start.keys = _key_resp_story_start_allKeys[-1].name  # just the last key pressed
                        key_resp_story_start.rt = _key_resp_story_start_allKeys[-1].rt
                        key_resp_story_start.duration = _key_resp_story_start_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in story_idComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "story_id" ---
            for thisComponent in story_idComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('story_id.stopped', globalClock.getTime(format='float'))
            # check responses
            if key_resp_story_start.keys in ['', [], None]:  # No response was made
                key_resp_story_start.keys = None
            go_back.addData('key_resp_story_start.keys',key_resp_story_start.keys)
            if key_resp_story_start.keys != None:  # we had a response
                go_back.addData('key_resp_story_start.rt', key_resp_story_start.rt)
                go_back.addData('key_resp_story_start.duration', key_resp_story_start.duration)
            # the Routine "story_id" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "blank_200ms" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('blank_200ms.started', globalClock.getTime(format='float'))
            # keep track of which components have finished
            blank_200msComponents = []
            for thisComponent in blank_200msComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "blank_200ms" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.2:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > .2-frameTolerance:
                    continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in blank_200msComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blank_200ms" ---
            for thisComponent in blank_200msComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('blank_200ms.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.200000)
            
            # --- Prepare to start Routine "wait" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('wait.started', globalClock.getTime(format='float'))
            # create starting attributes for key_go_back
            key_go_back.keys = []
            key_go_back.rt = []
            _key_go_back_allKeys = []
            # keep track of which components have finished
            waitComponents = [text_wait, key_go_back]
            for thisComponent in waitComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "wait" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 15.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > 15-frameTolerance:
                    continueRoutine = False
                
                # *text_wait* updates
                
                # if text_wait is starting this frame...
                if text_wait.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_wait.frameNStart = frameN  # exact frame index
                    text_wait.tStart = t  # local t and not account for scr refresh
                    text_wait.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_wait, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_wait.started')
                    # update status
                    text_wait.status = STARTED
                    text_wait.setAutoDraw(True)
                
                # if text_wait is active this frame...
                if text_wait.status == STARTED:
                    # update params
                    pass
                
                # if text_wait is stopping this frame...
                if text_wait.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_wait.tStartRefresh + 15-frameTolerance:
                        # keep track of stop time/frame for later
                        text_wait.tStop = t  # not accounting for scr refresh
                        text_wait.tStopRefresh = tThisFlipGlobal  # on global time
                        text_wait.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_wait.stopped')
                        # update status
                        text_wait.status = FINISHED
                        text_wait.setAutoDraw(False)
                
                # *key_go_back* updates
                waitOnFlip = False
                
                # if key_go_back is starting this frame...
                if key_go_back.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_go_back.frameNStart = frameN  # exact frame index
                    key_go_back.tStart = t  # local t and not account for scr refresh
                    key_go_back.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_go_back, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_go_back.started')
                    # update status
                    key_go_back.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_go_back.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_go_back.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_go_back is stopping this frame...
                if key_go_back.status == STARTED:
                    # is it time to stop? (based on local clock)
                    if tThisFlip > 15-frameTolerance:
                        # keep track of stop time/frame for later
                        key_go_back.tStop = t  # not accounting for scr refresh
                        key_go_back.tStopRefresh = tThisFlipGlobal  # on global time
                        key_go_back.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_go_back.stopped')
                        # update status
                        key_go_back.status = FINISHED
                        key_go_back.status = FINISHED
                if key_go_back.status == STARTED and not waitOnFlip:
                    theseKeys = key_go_back.getKeys(keyList=['left'], ignoreKeys=["escape"], waitRelease=False)
                    _key_go_back_allKeys.extend(theseKeys)
                    if len(_key_go_back_allKeys):
                        key_go_back.keys = _key_go_back_allKeys[-1].name  # just the last key pressed
                        key_go_back.rt = _key_go_back_allKeys[-1].rt
                        key_go_back.duration = _key_go_back_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                # Run 'Each Frame' code from go_back_code
                if key_go_back.keys == 'left':
                    go_back.finished = False
                else:
                    go_back.finished = True
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in waitComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "wait" ---
            for thisComponent in waitComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('wait.stopped', globalClock.getTime(format='float'))
            # check responses
            if key_go_back.keys in ['', [], None]:  # No response was made
                key_go_back.keys = None
            go_back.addData('key_go_back.keys',key_go_back.keys)
            if key_go_back.keys != None:  # we had a response
                go_back.addData('key_go_back.rt', key_go_back.rt)
                go_back.addData('key_go_back.duration', key_go_back.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-15.000000)
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'go_back'
        
        
        # --- Prepare to start Routine "stories" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('stories.started', globalClock.getTime(format='float'))
        story.setSound(file_names, secs=story_len, hamming=True)
        story.setVolume(1.0, log=False)
        story.seek(0)
        # keep track of which components have finished
        storiesComponents = [story, text_plus]
        for thisComponent in storiesComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "stories" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > story_len-frameTolerance:
                continueRoutine = False
            
            # if story is starting this frame...
            if story.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                story.frameNStart = frameN  # exact frame index
                story.tStart = t  # local t and not account for scr refresh
                story.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('story.started', tThisFlipGlobal)
                # update status
                story.status = STARTED
                story.play(when=win)  # sync with win flip
            
            # if story is stopping this frame...
            if story.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > story.tStartRefresh + story_len-frameTolerance:
                    # keep track of stop time/frame for later
                    story.tStop = t  # not accounting for scr refresh
                    story.tStopRefresh = tThisFlipGlobal  # on global time
                    story.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'story.stopped')
                    # update status
                    story.status = FINISHED
                    story.stop()
            # update story status according to whether it's playing
            if story.isPlaying:
                story.status = STARTED
            elif story.isFinished:
                story.status = FINISHED
            
            # *text_plus* updates
            
            # if text_plus is starting this frame...
            if text_plus.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text_plus.frameNStart = frameN  # exact frame index
                text_plus.tStart = t  # local t and not account for scr refresh
                text_plus.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_plus, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_plus.started')
                # update status
                text_plus.status = STARTED
                text_plus.setAutoDraw(True)
            
            # if text_plus is active this frame...
            if text_plus.status == STARTED:
                # update params
                pass
            
            # if text_plus is stopping this frame...
            if text_plus.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_plus.tStartRefresh + story_len-frameTolerance:
                    # keep track of stop time/frame for later
                    text_plus.tStop = t  # not accounting for scr refresh
                    text_plus.tStopRefresh = tThisFlipGlobal  # on global time
                    text_plus.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_plus.stopped')
                    # update status
                    text_plus.status = FINISHED
                    text_plus.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in storiesComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "stories" ---
        for thisComponent in storiesComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('stories.stopped', globalClock.getTime(format='float'))
        story.pause()  # ensure sound has stopped at end of Routine
        # the Routine "stories" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "blank_15" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('blank_15.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        blank_15Components = [blank_text]
        for thisComponent in blank_15Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank_15" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 15.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > 15-frameTolerance:
                continueRoutine = False
            
            # *blank_text* updates
            
            # if blank_text is starting this frame...
            if blank_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blank_text.frameNStart = frameN  # exact frame index
                blank_text.tStart = t  # local t and not account for scr refresh
                blank_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blank_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blank_text.started')
                # update status
                blank_text.status = STARTED
                blank_text.setAutoDraw(True)
            
            # if blank_text is active this frame...
            if blank_text.status == STARTED:
                # update params
                pass
            
            # if blank_text is stopping this frame...
            if blank_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blank_text.tStartRefresh + 15-frameTolerance:
                    # keep track of stop time/frame for later
                    blank_text.tStop = t  # not accounting for scr refresh
                    blank_text.tStopRefresh = tThisFlipGlobal  # on global time
                    blank_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blank_text.stopped')
                    # update status
                    blank_text.status = FINISHED
                    blank_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank_15Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank_15" ---
        for thisComponent in blank_15Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('blank_15.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-15.000000)
        
        # --- Prepare to start Routine "questions" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('questions.started', globalClock.getTime(format='float'))
        question_text.setText(question)
        # create starting attributes for question_resp
        question_resp.keys = []
        question_resp.rt = []
        _question_resp_allKeys = []
        # keep track of which components have finished
        questionsComponents = [question_text, question_resp]
        for thisComponent in questionsComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "questions" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *question_text* updates
            
            # if question_text is starting this frame...
            if question_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                question_text.frameNStart = frameN  # exact frame index
                question_text.tStart = t  # local t and not account for scr refresh
                question_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(question_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'question_text.started')
                # update status
                question_text.status = STARTED
                question_text.setAutoDraw(True)
            
            # if question_text is active this frame...
            if question_text.status == STARTED:
                # update params
                pass
            
            # *question_resp* updates
            waitOnFlip = False
            
            # if question_resp is starting this frame...
            if question_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                question_resp.frameNStart = frameN  # exact frame index
                question_resp.tStart = t  # local t and not account for scr refresh
                question_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(question_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'question_resp.started')
                # update status
                question_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(question_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(question_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if question_resp.status == STARTED and not waitOnFlip:
                theseKeys = question_resp.getKeys(keyList=['right','space','enter'], ignoreKeys=["escape"], waitRelease=False)
                _question_resp_allKeys.extend(theseKeys)
                if len(_question_resp_allKeys):
                    question_resp.keys = _question_resp_allKeys[-1].name  # just the last key pressed
                    question_resp.rt = _question_resp_allKeys[-1].rt
                    question_resp.duration = _question_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in questionsComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "questions" ---
        for thisComponent in questionsComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('questions.stopped', globalClock.getTime(format='float'))
        # check responses
        if question_resp.keys in ['', [], None]:  # No response was made
            question_resp.keys = None
        story_loop.addData('question_resp.keys',question_resp.keys)
        if question_resp.keys != None:  # we had a response
            story_loop.addData('question_resp.rt', question_resp.rt)
            story_loop.addData('question_resp.duration', question_resp.duration)
        # the Routine "questions" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "blank_200ms" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('blank_200ms.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        blank_200msComponents = []
        for thisComponent in blank_200msComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank_200ms" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.2:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > .2-frameTolerance:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank_200msComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank_200ms" ---
        for thisComponent in blank_200msComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('blank_200ms.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.200000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'story_loop'
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
