Inputs: 

ORB\_Start\_H\_NY(8), 

ORB\_Start\_M\_NY(30), 

ORB\_Dur\_Min(30), 

Trade\_End\_H\_NY(14), 

Trade\_End\_M\_NY(0), 

TP\_RR\_Ratio(1.0), 

Num\_Contracts(1), 

DeltaThreshold(200), 

UseCumulativeDelta(false), 

CumDeltaThreshold(500); 

Variables: 

ORB\_Start\_Time(0), ORB\_End\_Time(0), Trade\_End\_Time(0), 

ORB\_Inited(false), ORB\_Session\_Ended(false), 

ORB\_High(0), ORB\_Low(0), 

Long\_Done\_Today(false), Was\_Long(false), 

Pos\_SL(0), Pos\_TP(0), 

Cur\_Date\_Int(0), Prev\_Bar\_Time(0), Bar\_Time(0), 

In\_ORB(false), Trade\_Active(false), 

EP(0), SL\_Price(0), TP\_Price(0), 

BarDelta(0), CumDelta(0), DeltaOK(false); 

{ \===== ONE-TIME SETUP \===== } 

Once begin 

ORB\_Start\_Time \= ORB\_Start\_H\_NY \* 100 \+ ORB\_Start\_M\_NY; 

Value1 \= IntPortion(ORB\_Start\_Time / 100); 

Value2 \= Mod(ORB\_Start\_Time, 100); 

Value3 \= Value1 \* 60 \+ Value2 \+ ORB\_Dur\_Min; 

Value3 \= Mod(Value3, 24 \* 60); 

if Value3 \< 0 then Value3 \= Value3 \+ 24 \* 60; 

ORB\_End\_Time \= IntPortion(Value3 / 60\) \* 100 \+ Mod(Value3, 60); 

Trade\_End\_Time \= Trade\_End\_H\_NY \* 100 \+ Trade\_End\_M\_NY; 

end; 

{ \===== SKIP FIRST BAR \===== } 

if CurrentBar \< 2 then begin 

Cur\_Date\_Int \= Date; Prev\_Bar\_Time \= Time; 

end; 

if CurrentBar \>= 2 then begin 

{ \===== NEW DAY RESET \===== } 

if Date \<\> Cur\_Date\_Int then begin 

Cur\_Date\_Int \= Date; 

ORB\_Inited \= false; ORB\_Session\_Ended \= false; 

ORB\_High \= 0; ORB\_Low \= 0; 

Long\_Done\_Today \= false; Was\_Long \= false; 

Pos\_SL \= 0; Pos\_TP \= 0; 

CumDelta \= 0; 

end; 

Bar\_Time \= Time; 

In\_ORB \= (Bar\_Time \> ORB\_Start\_Time) and (Bar\_Time \<= ORB\_End\_Time); 

Trade\_Active \= (Bar\_Time \> ORB\_End\_Time) and (Bar\_Time \<= Trade\_End\_Time); 

{ \===== DELTA CALCULATION \===== } 

BarDelta \= Upticks \- Downticks; 

if Bar\_Time \> ORB\_Start\_Time and Bar\_Time \<= Trade\_End\_Time then 

CumDelta \= CumDelta \+ BarDelta; 

{ continued on next page... }  
{ ...continued from previous page } 

{ \===== BUILD ORB \===== } 

if In\_ORB then begin 

if ORB\_Inited \= false then begin 

ORB\_High \= High; ORB\_Low \= Low; ORB\_Inited \= true; 

end 

else begin 

if High \> ORB\_High then ORB\_High \= High; 

if Low \< ORB\_Low then ORB\_Low \= Low; 

end; 

end; 

if ORB\_Session\_Ended \= false and ORB\_Inited and Bar\_Time \> ORB\_End\_Time then ORB\_Session\_Ended \= true; 

{ \===== DETECT LONG \-\> FLAT TO CLEAR LEVELS \===== } 

if MarketPosition \= 1 then Was\_Long \= true; 

if MarketPosition \= 0 and Was\_Long then begin 

Was\_Long \= false; Pos\_SL \= 0; Pos\_TP \= 0; 

end; 

{ \===== EOD EXIT \===== } 

if Bar\_Time \>= Trade\_End\_Time and Prev\_Bar\_Time \< Trade\_End\_Time then begin if MarketPosition \= 1 then   
Sell ("EOD-L") this bar on close; 

end; 

Prev\_Bar\_Time \= Bar\_Time; 

{ \===== TP / SL WHILE LONG \===== } 

if MarketPosition \= 1 and Pos\_SL \<\> 0 and Pos\_TP \<\> 0 then begin Sell ("L-TP") next bar at Pos\_TP limit;   
Sell ("L-SL") next bar at Pos\_SL stop; 

end; 

{ \===== ENTRY LOGIC \===== } 

if ORB\_Session\_Ended 

and Trade\_Active 

and Long\_Done\_Today \= false 

and MarketPosition \= 0 

and Close \> ORB\_High then begin 

EP \= Close; SL\_Price \= ORB\_Low; 

if SL\_Price \< EP then begin 

DeltaOK \= BarDelta \>= DeltaThreshold; 

if UseCumulativeDelta then 

DeltaOK \= DeltaOK and (CumDelta \>= CumDeltaThreshold); 

if DeltaOK then begin 

TP\_Price \= EP \+ TP\_RR\_Ratio \* (EP \- SL\_Price); 

Pos\_SL \= SL\_Price; Pos\_TP \= TP\_Price; 

Buy ("ORB-L MKT") Num\_Contracts contracts this bar on close; Long\_Done\_Today \= true; 

end; 

end; 

end; 

end;