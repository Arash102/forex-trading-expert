//+------------------------------------------------------------------+
//| DEBCO_ChartEventHelper.mq5                                       |
//| Chart marker + screenshot helper only. No strategy calculations. |
//+------------------------------------------------------------------+
#property strict
#property version   "0.1.13"
#property description "Reads DEBCO .cmd chart events, draws setup labels/green circles, and saves screenshots."

input string EventFolder = "debco_chart_events";   // MQL5/Files/<EventFolder>
input int TimerSeconds = 1;
input int ScreenshotWidth = 1280;
input int ScreenshotHeight = 720;

int OnInit()
{
   EventSetTimer(TimerSeconds);
   Print("DEBCO ChartEventHelper started for symbol ", _Symbol);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTimer()
{
   ProcessEventFiles();
}

string StripUTC(string s)
{
   StringReplace(s, "T", " ");
   StringReplace(s, "Z", "");
   return s;
}

void ProcessEventFiles()
{
   string pattern = EventFolder + "\\*.cmd";
   string filename;
   long handle = FileFindFirst(pattern, filename, FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return;

   do
   {
      string full = EventFolder + "\\" + filename;
      ProcessOneFile(full);
   }
   while(FileFindNext(handle, filename));

   FileFindClose(handle);
}

void ProcessOneFile(string relative_path)
{
   int fh = FileOpen(relative_path, FILE_READ | FILE_TXT | FILE_COMMON | FILE_ANSI);
   if(fh == INVALID_HANDLE)
      return;
   string line = FileReadString(fh);
   FileClose(fh);

   string parts[];
   int n = StringSplit(line, ';', parts);
   if(n < 11)
      return;

   string event_id = parts[0];
   string event_type = parts[1];
   string symbol = parts[2];
   string setup_id = parts[3];
   string side = parts[4];
   string magic = parts[5];
   string time_utc = parts[6];
   string price_txt = parts[7];
   string marker_color = parts[8];
   string label = parts[9];
   string screenshot_name = parts[10];

   if(symbol != _Symbol)
      return;

   datetime t = StringToTime(StripUTC(time_utc));
   double price = StringToDouble(price_txt);
   if(price <= 0.0)
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   color c = clrLime;
   if(event_type == "exit") c = clrRed;

   string base = "DEBCO_" + event_id + "_" + setup_id;
   string circle_name = base + "_circle";
   string text_name = base + "_label";

   ObjectCreate(0, circle_name, OBJ_ARROW, 0, t, price);
   ObjectSetInteger(0, circle_name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, circle_name, OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, circle_name, OBJPROP_ARROWCODE, 159); // filled circle-like marker

   ObjectCreate(0, text_name, OBJ_TEXT, 0, t, price);
   ObjectSetString(0, text_name, OBJPROP_TEXT, setup_id);
   ObjectSetInteger(0, text_name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, text_name, OBJPROP_FONTSIZE, 8);
   ObjectSetInteger(0, text_name, OBJPROP_ANCHOR, ANCHOR_LEFT_LOWER);

   if(StringLen(screenshot_name) > 0)
   {
      ChartScreenShot(0, screenshot_name, ScreenshotWidth, ScreenshotHeight, ALIGN_RIGHT);
   }

   string done = relative_path + ".done";
   FileMove(relative_path, FILE_COMMON, done, FILE_COMMON | FILE_REWRITE);
}
