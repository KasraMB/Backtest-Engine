//@version=6
indicator('AnchoredMeanReversion Strategy By K', overlay = true)

// === Inputs ===
show930       = input.bool(true,  'Session 1',                                  group = 'Session Opens')
start930      = input.int(930,    'S1 Start (HHMM)',                            group = 'Session Opens', minval = 0, maxval = 2359)
end930        = input.int(1100,   'S1 End (HHMM)',                              group = 'Session Opens', minval = 0, maxval = 2359)

show1400      = input.bool(true,  'Session 2',                                  group = 'Session Opens')
start1400     = input.int(1400,   'S2 Start (HHMM)',                            group = 'Session Opens', minval = 0, maxval = 2359)
end1400       = input.int(1530,   'S2 End (HHMM)',                              group = 'Session Opens', minval = 0, maxval = 2359)

show2000      = input.bool(false, 'Session 3',                                  group = 'Session Opens')
start2000     = input.int(2000,   'S3 Start (HHMM)',                            group = 'Session Opens', minval = 0, maxval = 2359)
end2000       = input.int(2130,   'S3 End (HHMM)',                              group = 'Session Opens', minval = 0, maxval = 2359)

show300       = input.bool(false, 'Session 4',                                  group = 'Session Opens')
start300      = input.int(300,    'S4 Start (HHMM)',                            group = 'Session Opens', minval = 0, maxval = 2359)
end300        = input.int(430,    'S4 End (HHMM)',                              group = 'Session Opens', minval = 0, maxval = 2359)

link1400To930 = input.bool(false, 'Link S2 to S1 fair price',                  group = 'Session Opens')

showTrades         = input.bool(true,  'Show Trade Setups',                    group = 'Trade Setups')
requireBOS         = input.bool(true,  'Require BOS for Signal',               group = 'Trade Setups')
filterTPBeyondFair = input.bool(false, 'Filter by min entry → fair distance',  group = 'Trade Setups')
tpBeyondFairPct    = input.float(50.0, 'Min % of TP range between entry and fair', group = 'Trade Setups', minval = 0.0, maxval = 100.0, step = 1.0)
moveSLToEntry      = input.bool(false, 'Move SL to Entry at Fair Price',       group = 'Trade Setups')
skipFirstMins      = input.int(0,      'Skip First N Minutes of Window',       group = 'Trade Setups', minval = 0)
againstFairMins    = input.int(15,     'Allow Against-Fair Trades (mins)',     group = 'Trade Setups', minval = 0)
tpMultiple         = input.float(1.5,  'TP Multiple',                          group = 'Trade Setups', minval = 0.1, step = 0.1)
atrSlLow           = input.float(7.0,  'ATR Lower Threshold',                  group = 'Trade Setups', minval = 0.0)
atrSlHigh          = input.float(20.0, 'ATR Upper Threshold',                  group = 'Trade Setups', minval = 0.0)
slTicksLow         = input.int(66,     'SL Ticks (ATR < lower)',               group = 'Trade Setups', minval = 1)
slTicksMid         = input.int(100,    'SL Ticks (lower <= ATR < upper)',      group = 'Trade Setups', minval = 1)
slTicksHigh        = input.int(200,    'SL Ticks (ATR >= upper)',              group = 'Trade Setups', minval = 1)
riskColor          = input.color(color.new(color.red,   70), 'Risk Area Color',   group = 'Trade Setups')
rewardColor        = input.color(color.new(color.green, 70), 'Reward Area Color', group = 'Trade Setups')
entryLineCol       = input.color(color.white,                 'Entry Line Color',  group = 'Trade Setups')
maxSetups          = input.int(30, 'Max Setups To Show', group = 'Trade Setups', minval = 1)

n         = input.int(1,    'Swing Periods',                minval = 1,                              group = 'Base Signal')
thresholdMW = input.float(0.15,'Wick % (Marubozu)',     minval = 0.0, maxval = 1.0, step = 0.01, group = 'Base Signal')
thresholdUW = input.float(0.8,'Close Position Threshold (Upper Wick)',     minval = 0.0, maxval = 1.0, step = 0.01, group = 'Base Signal')
atrLen    = input.int(14,   'ATR Length',                   minval = 1,                              group = 'Base Signal')
atrMult   = input.float(0.0,'Min ATR Multiplier (0 = off)', minval = 0.0, step = 0.1,               group = 'Base Signal')
displacementStyle = input.string("Upper Wick", "Displacement Detection Style", options = ["Upper Wick", "Marubozu Style"], group="Base Signal")

showBOSLines = input.bool(true, 'Show BOS Lines',        group = 'BOS Lines')
maxBOSLines  = input.int(5,     'Max BOS Lines Visible', group = 'BOS Lines', minval = 1)

// === Constants ===
TZ         = 'America/New_York'
MS_PER_MIN = 60 * 1000

// === HHMM helpers ===
nowHHMM = hour(time, TZ) * 100 + minute(time, TZ)
inWindow(s, e) => nowHHMM >= s and nowHHMM < e
isOpen(s)      => nowHHMM == s

// === ATR ===
atrValue    = ta.atr(atrLen)
candleRange = high - low
atrFilter   = atrMult == 0.0 ? true : candleRange >= atrValue * atrMult

// === Displacement candle ===
upperWick   = high - math.max(open, close)
lowerWick   = math.min(open, close) - low

totalWick = upperWick + lowerWick

rangeSafe = candleRange == 0 ? 0.01 :  candleRange

marubozu =  totalWick / rangeSafe <= thresholdMW


bullRange = high - open
bullPos   = bullRange != 0 ? (close - open) / bullRange : 0.0
bullCond  = close >= open and bullPos >= thresholdUW

bearRange = open - low
bearPos   = bearRange != 0 ? (open - close) / bearRange : 0.0
bearCond  = close < open and bearPos >= thresholdUW
isDisplacement = false
if displacementStyle == 'Upper Wick'
    isDisplacement := (bullCond or bearCond) and atrFilter
else
    isDisplacement := marubozu and atrFilter
// === Swing detection ===
bool isHH = true
for i = 1 to n
    isHH := isHH and high[n] > high[n - i] and high[n] > high[n + i]

bool isLL = true
for i = 1 to n
    isLL := isLL and low[n] < low[n - i] and low[n] < low[n + i]

plotshape(isHH, style = shape.triangleup,   location = location.abovebar, offset = -n, color = #009688, size = size.tiny)
plotshape(isLL, style = shape.triangledown, location = location.belowbar, offset = -n, color = #F44336, size = size.tiny)

// === Swing arrays ===
var highPrices = array.new_float()
var highBars   = array.new_int()
var highBroken = array.new_bool()
var lowPrices  = array.new_float()
var lowBars    = array.new_int()
var lowBroken  = array.new_bool()

if isHH
    newHigh = high[n]
    i = array.size(highPrices) - 1
    while i >= 0
        if not array.get(highBroken, i) and array.get(highPrices, i) < newHigh
            array.remove(highPrices, i)
            array.remove(highBars, i)
            array.remove(highBroken, i)
        i -= 1
    array.push(highPrices, newHigh)
    array.push(highBars,   bar_index - n)
    array.push(highBroken, false)

if isLL
    newLow = low[n]
    i = array.size(lowPrices) - 1
    while i >= 0
        if not array.get(lowBroken, i) and array.get(lowPrices, i) > newLow
            array.remove(lowPrices, i)
            array.remove(lowBars, i)
            array.remove(lowBroken, i)
        i -= 1
    array.push(lowPrices, newLow)
    array.push(lowBars,   bar_index - n)
    array.push(lowBroken, false)

// === BOS line arrays ===
var array<line> bosHighLines = array.new<line>()
var array<line> bosLowLines  = array.new<line>()

// === Check highs ===
bool brokeHigh = false
if array.size(highPrices) > 0
    int bestIdx = na
    for i = 0 to array.size(highPrices) - 1
        if not array.get(highBroken, i)
            lvl = array.get(highPrices, i)
            if close > lvl
                bestIdx := na(bestIdx) ? i : (lvl > array.get(highPrices, bestIdx) ? i : bestIdx)
    if not na(bestIdx)
        lvl = array.get(highPrices, bestIdx)
        if showBOSLines
            newLine = line.new(array.get(highBars, bestIdx), lvl, bar_index, lvl, color = color.green, width = 2)
            array.push(bosHighLines, newLine)
            while array.size(bosHighLines) > maxBOSLines
                line.delete(array.shift(bosHighLines))
        brokeHigh := true
        for i = 0 to array.size(highPrices) - 1
            if not array.get(highBroken, i) and close > array.get(highPrices, i)
                array.set(highBroken, i, true)

// === Check lows ===
bool brokeLow = false
if array.size(lowPrices) > 0
    int bestIdx = na
    for i = 0 to array.size(lowPrices) - 1
        if not array.get(lowBroken, i)
            lvl = array.get(lowPrices, i)
            if close < lvl
                bestIdx := na(bestIdx) ? i : (lvl < array.get(lowPrices, bestIdx) ? i : bestIdx)
    if not na(bestIdx)
        lvl = array.get(lowPrices, bestIdx)
        if showBOSLines
            newLine = line.new(array.get(lowBars, bestIdx), lvl, bar_index, lvl, color = color.red, width = 2)
            array.push(bosLowLines, newLine)
            while array.size(bosLowLines) > maxBOSLines
                line.delete(array.shift(bosLowLines))
        brokeLow := true
        for i = 0 to array.size(lowPrices) - 1
            if not array.get(lowBroken, i) and close < array.get(lowPrices, i)
                array.set(lowBroken, i, true)

barcolor(isDisplacement and (brokeHigh or brokeLow) ? color.white : na)

// === Session open detection ===
is930  = show930  and isOpen(start930)
is1400 = show1400 and isOpen(start1400)
is2000 = show2000 and isOpen(start2000)
is300  = show300  and isOpen(start300)

// === Session open lines and labels ===
var line  l930   = na
var label lb930  = na
var line  l1400  = na
var label lb1400 = na
var line  l2000  = na
var label lb2000 = na
var line  l300   = na
var label lb300  = na

if is930
    if not na(l930)
        line.delete(l930)
        label.delete(lb930)
    l930  := line.new(bar_index, open, bar_index + 1, open, color = color.new(color.yellow, 0), width = 1, style = line.style_dashed)
    lb930 := label.new(bar_index, open, 'S1 fair price', xloc = xloc.bar_index, yloc = yloc.price, style = label.style_label_left, color = color.new(color.yellow, 80), textcolor = color.yellow, size = size.small)

if is1400
    if not na(l1400)
        line.delete(l1400)
        label.delete(lb1400)
    l1400  := line.new(bar_index, open, bar_index + 1, open, color = color.new(color.orange, 0), width = 1, style = line.style_dashed)
    lb1400 := label.new(bar_index, open, 'S2 fair price', xloc = xloc.bar_index, yloc = yloc.price, style = label.style_label_left, color = color.new(color.orange, 80), textcolor = color.orange, size = size.small)

if is2000
    if not na(l2000)
        line.delete(l2000)
        label.delete(lb2000)
    l2000  := line.new(bar_index, open, bar_index + 1, open, color = color.new(color.purple, 0), width = 1, style = line.style_dashed)
    lb2000 := label.new(bar_index, open, 'S3 fair price', xloc = xloc.bar_index, yloc = yloc.price, style = label.style_label_left, color = color.new(color.purple, 80), textcolor = color.purple, size = size.small)

if is300
    if not na(l300)
        line.delete(l300)
        label.delete(lb300)
    l300  := line.new(bar_index, open, bar_index + 1, open, color = color.new(color.aqua, 0), width = 1, style = line.style_dashed)
    lb300 := label.new(bar_index, open, 'S4 fair price', xloc = xloc.bar_index, yloc = yloc.price, style = label.style_label_left, color = color.new(color.aqua, 80), textcolor = color.aqua, size = size.small)

if barstate.islast
    if show930  and not na(lb930)
        label.set_x(lb930,  bar_index)
        line.set_x2(l930,   bar_index)
    if show1400 and not na(lb1400)
        label.set_x(lb1400, bar_index)
        line.set_x2(l1400,  bar_index)
    if show2000 and not na(lb2000)
        label.set_x(lb2000, bar_index)
        line.set_x2(l2000,  bar_index)
    if show300  and not na(lb300)
        label.set_x(lb300,  bar_index)
        line.set_x2(l300,   bar_index)

// === Session fair prices ===
var float last930Price  = na
var float last1400Price = na
var float last2000Price = na
var float last300Price  = na
var int   last930Time   = na
var int   last1400Time  = na
var int   last2000Time  = na
var int   last300Time   = na

if is930
    last930Price := open
    last930Time  := time
if is1400
    last1400Price := open
    last1400Time  := time
if is2000
    last2000Price := open
    last2000Time  := time
if is300
    last300Price := open
    last300Time  := time

// === Window, fair price and direction tracking ===
againstFairMs = againstFairMins * MS_PER_MIN
skipFirstMs   = skipFirstMins   * MS_PER_MIN

withinWindow    = false
var float activeFairPrice = na
var int   activeOpenTime  = na

if show930  and not na(last930Price)  and inWindow(start930,  end930)
    withinWindow    := true
    activeFairPrice := last930Price
    activeOpenTime  := last930Time

if show1400 and not na(last1400Price) and inWindow(start1400, end1400)
    withinWindow    := true
    activeFairPrice := link1400To930 ? last930Price : last1400Price
    activeOpenTime  := link1400To930 ? na : last1400Time

if show2000 and not na(last2000Price) and inWindow(start2000, end2000)
    withinWindow    := true
    activeFairPrice := last2000Price
    activeOpenTime  := last2000Time

if show300  and not na(last300Price)  and inWindow(start300,  end300)
    withinWindow    := true
    activeFairPrice := last300Price
    activeOpenTime  := last300Time

// === Direction filter ===
elapsedMs      = withinWindow and not na(activeOpenTime) ? time - activeOpenTime : na
inFirstWindow  = not na(elapsedMs) and elapsedMs <= againstFairMs
pastSkipWindow = not na(elapsedMs) and elapsedMs >= skipFirstMs

towardsFairLong  = not na(activeFairPrice) and close < activeFairPrice
towardsFairShort = not na(activeFairPrice) and close > activeFairPrice

longDirectionOK  = towardsFairLong  or inFirstWindow
shortDirectionOK = towardsFairShort or inFirstWindow

// === SL / TP ===
tickSize   = syminfo.mintick
slTicks    = atrValue < atrSlLow ? slTicksLow : atrValue < atrSlHigh ? slTicksMid : slTicksHigh
slDistance = slTicks * tickSize
tpDistance = slDistance * tpMultiple

// === TP beyond fair price filter ===
longTPBeyondFairOK  = true
shortTPBeyondFairOK = true

if filterTPBeyondFair and not na(activeFairPrice)
    longTP  = close + tpDistance
    shortTP = close - tpDistance

    if towardsFairLong and longTP > activeFairPrice
        beyondRatio        = (activeFairPrice - close) / tpDistance
        longTPBeyondFairOK := beyondRatio >= (tpBeyondFairPct / 100.0)

    if towardsFairShort and shortTP < activeFairPrice
        beyondRatio         = (close - activeFairPrice) / tpDistance
        shortTPBeyondFairOK := beyondRatio >= (tpBeyondFairPct / 100.0)

// === Position tracking ===
var bool  positionActive = false
var bool  isLong         = false
var float activeSL       = na
var float activeTP       = na
var bool  closedThisBar  = false

closedThisBar := false

if positionActive and not na(activeSL) and not na(activeTP)
    if isLong
        if low <= activeSL or high >= activeTP
            positionActive := false
            activeSL       := na
            activeTP       := na
            closedThisBar  := true
    else
        if high >= activeSL or low <= activeTP
            positionActive := false
            activeSL       := na
            activeTP       := na
            closedThisBar  := true

// === Signals ===
longSignal  = showTrades and not positionActive and not closedThisBar and pastSkipWindow and isDisplacement and bullCond and atrFilter and (requireBOS ? (brokeHigh) : true) and withinWindow and longDirectionOK  and longTPBeyondFairOK
shortSignal = showTrades and not positionActive and not closedThisBar and pastSkipWindow and isDisplacement and bearCond and atrFilter and (requireBOS ? (brokeLow) : true) and withinWindow and shortDirectionOK and shortTPBeyondFairOK

// === Entry price tracking ===
var float entryPrice     = na
var bool  slMovedToEntry = false
var bool meanRev = false

if longSignal
    positionActive := true
    isLong         := true
    entryPrice     := close
    activeSL       := close - slDistance
    activeTP       := close + tpDistance
    slMovedToEntry := false
    if activeFairPrice > entryPrice
        meanRev := true

if shortSignal
    positionActive := true
    isLong         := false
    entryPrice     := close
    activeSL       := close + slDistance
    activeTP       := close - tpDistance
    slMovedToEntry := false
    if activeFairPrice < entryPrice
        meanRev := true

// === Move SL to entry when price hits fair price (reversion trades only) ===
fairHit = meanRev and moveSLToEntry and positionActive and not slMovedToEntry and not inFirstWindow and not na(activeFairPrice) and not na(entryPrice) and ((isLong and high >= activeFairPrice) or (not isLong and low <= activeFairPrice))

if fairHit
    activeSL       := entryPrice
    slMovedToEntry := true

// === Trade drawing arrays ===
var array<line> entryLines  = array.new<line>()
var array<box>  riskBoxes   = array.new<box>()
var array<box>  rewardBoxes = array.new<box>()

if longSignal
    ep = close
    sl = ep - slDistance
    tp = ep + tpDistance
    array.push(entryLines,  line.new(bar_index, ep, bar_index, ep, color = entryLineCol, width = 2))
    array.push(riskBoxes,   box.new(bar_index,  ep, bar_index, sl, bgcolor = riskColor,   border_color = na))
    array.push(rewardBoxes, box.new(bar_index,  ep, bar_index, tp, bgcolor = rewardColor, border_color = na))

if shortSignal
    ep = close
    sl = ep + slDistance
    tp = ep - tpDistance
    array.push(entryLines,  line.new(bar_index, ep, bar_index, ep, color = entryLineCol, width = 2))
    array.push(riskBoxes,   box.new(bar_index,  ep, bar_index, sl, bgcolor = riskColor,   border_color = na))
    array.push(rewardBoxes, box.new(bar_index,  ep, bar_index, tp, bgcolor = rewardColor, border_color = na))

// === Extend active drawing each bar, snap shut when closed ===
if array.size(entryLines) > 0 and positionActive
    lastEntry  = array.get(entryLines,  array.size(entryLines)  - 1)
    lastRisk   = array.get(riskBoxes,   array.size(riskBoxes)   - 1)
    lastReward = array.get(rewardBoxes, array.size(rewardBoxes) - 1)
    line.set_x2(lastEntry,    bar_index)
    box.set_right(lastRisk,   bar_index)
    box.set_right(lastReward, bar_index)

// === Cleanup old setups ===
while array.size(entryLines) > maxSetups
    line.delete(array.shift(entryLines))
while array.size(riskBoxes) > maxSetups
    box.delete(array.shift(riskBoxes))
while array.size(rewardBoxes) > maxSetups
    box.delete(array.shift(rewardBoxes))

plotshape(longSignal,  style = shape.arrowup,   location = location.belowbar, color = color.green,  size = size.normal, title = 'Long Signal')
plotshape(shortSignal, style = shape.arrowdown, location = location.abovebar, color = color.red,    size = size.normal, title = 'Short Signal')
plotshape(fairHit,     style = shape.arrowup,   location = location.belowbar, color = color.yellow, size = size.small,  title = 'SL Moved to Entry')