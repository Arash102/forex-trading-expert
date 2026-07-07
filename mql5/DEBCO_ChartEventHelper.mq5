//+------------------------------------------------------------------+
//| DEBCO_ChartEventHelper.mq5                                       |
//| Chart marker + screenshot helper only. No strategy calculations. |
//+------------------------------------------------------------------+
#property strict
#property version   "0.1.13g4"
#property description "Reads DEBCO .txt chart events, draws setup labels, and saves screenshots."

input string EventFolder = "debco_chart_events";       // Common/Files/<EventFolder>
input string ScreenshotFolder = "DEBCO_Screenshots";   // MQL5/Files/<ScreenshotFolder>
input int TimerSeconds = 1;
input int ScreenshotWidth = 1280;
input int ScreenshotHeight = 720;
input bool TakeScreenshots = true;
input bool DrawLabels = true;
input int LabelFontSize = 9;
input bool DebugScan = true;

int OnInit()
{
   EventSetTimer(TimerSeconds);
   FolderCreate(ScreenshotFolder);

   Print("DEBCO ChartEventHelper started. Version=0.1.13g4 Symbol=", _Symbol,
         " EventFolder(Common)=", EventFolder,
         " ScreenshotFolder(MQL5/Files)=", ScreenshotFolder,
         " CommonDataPath=", TerminalInfoString(TERMINAL_COMMONDATA_PATH));

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

bool EndsWith(string text, string suffix)
{
   int lt = StringLen(text);
   int ls = StringLen(suffix);

   if(ls > lt)
      return false;

   return StringSubstr(text, lt - ls, ls) == suffix;
}

bool SymbolMatches(string event_symbol)
{
   if(event_symbol == _Symbol)
      return true;

   if(StringFind(_Symbol, event_symbol) >= 0)
      return true;

   if(StringFind(event_symbol, _Symbol) >= 0)
      return true;

   return false;
}

bool FileNameLooksForThisChart(string filename)
{
   if(StringFind(filename, _Symbol) >= 0)
      return true;

   if(StringFind(_Symbol, "EURUSD") >= 0 && StringFind(filename, "EURUSD") >= 0)
      return true;

   if(StringFind(_Symbol, "XAUUSD") >= 0 && StringFind(filename, "XAUUSD") >= 0)
      return true;

   return false;
}

string StripUTC(string s)
{
   StringReplace(s, "T", " ");
   StringReplace(s, "Z", "");
   return s;
}

string SafeFileName(string s)
{
   StringReplace(s, ":", "");
   StringReplace(s, "-", "");
   StringReplace(s, " ", "_");
   StringReplace(s, "/", "_");
   StringReplace(s, "\\", "_");
   return s;
}

void ProcessEventFiles()
{
   string pattern = EventFolder + "\\*";
   string filename = "";

   ResetLastError();
   long handle = FileFindFirst(pattern, filename, FILE_COMMON);

   if(handle == INVALID_HANDLE)
   {
      if(DebugScan)
         Print("DEBCO scan: no files found. Pattern(Common)=", pattern, " error=", GetLastError());
      return;
   }

   string event_files[];
   int total_files = 0;
   int event_count = 0;

   do
   {
      total_files++;

      if(!EndsWith(filename, ".txt"))
         continue;

      if(!FileNameLooksForThisChart(filename))
      {
         if(DebugScan)
            Print("DEBCO scan: skip txt not for this chart. file=", filename, " chart_symbol=", _Symbol);
         continue;
      }

      ArrayResize(event_files, event_count + 1);
      event_files[event_count] = EventFolder + "\\" + filename;
      event_count++;
   }
   while(FileFindNext(handle, filename));

   FileFindClose(handle);

   if(DebugScan)
      Print("DEBCO scan done. Pattern=", pattern,
            " files=", total_files,
            " txt_for_chart=", event_count,
            " chart_symbol=", _Symbol);

   for(int i = 0; i < event_count; i++)
   {
      if(DebugScan)
         Print("DEBCO processing txt=", event_files[i], " chart_symbol=", _Symbol);

      ProcessOneFile(event_files[i]);
   }
}

void ProcessOneFile(string relative_path)
{
   ResetLastError();

   int flags = FILE_READ | FILE_TXT | FILE_COMMON | FILE_ANSI | FILE_SHARE_READ | FILE_SHARE_WRITE;

   int fh = FileOpen(relative_path, flags);

   if(fh == INVALID_HANDLE)
   {
      Print("DEBCO: cannot open event file: ", relative_path, " error=", GetLastError());
      return;
   }

   string line = FileReadString(fh);
   FileClose(fh);

   if(StringLen(line) <= 0)
   {
      Print("DEBCO: empty event file: ", relative_path);
      return;
   }

   string parts[];
   int n = StringSplit(line, ';', parts);

   if(n < 11)
   {
      Print("DEBCO: invalid event format: ", relative_path, " parts=", n, " line=", line);
      return;
   }

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

   if(!SymbolMatches(symbol))
   {
      if(DebugScan)
         Print("DEBCO: skip event for symbol=", symbol, " on chart=", _Symbol, " file=", relative_path);
      return;
   }

   datetime t = StringToTime(StripUTC(time_utc));
   double price = StringToDouble(price_txt);

   if(price <= 0.0)
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   color marker_clr = clrLime;

   if(event_type == "exit")
      marker_clr = clrRed;

   string base_name = "DEBCO_" + event_id + "_" + setup_id;
   string circle_name = base_name + "_circle";
   string text_name = base_name + "_label";

   ObjectDelete(0, circle_name);
   ObjectDelete(0, text_name);

   bool circle_ok = ObjectCreate(0, circle_name, OBJ_ARROW, 0, t, price);
   if(circle_ok)
   {
      ObjectSetInteger(0, circle_name, OBJPROP_COLOR, marker_clr);
      ObjectSetInteger(0, circle_name, OBJPROP_WIDTH, 2);
      ObjectSetInteger(0, circle_name, OBJPROP_ARROWCODE, 159);
   }
   else
   {
      Print("DEBCO: ObjectCreate circle failed. error=", GetLastError());
   }

   if(DrawLabels)
   {
      string label_text = setup_id + " | " + side + " | magic=" + magic;

      if(event_type == "exit")
         label_text = "EXIT | " + label_text;

      bool text_ok = ObjectCreate(0, text_name, OBJ_TEXT, 0, t, price);

      if(text_ok)
      {
         ObjectSetString(0, text_name, OBJPROP_TEXT, label_text);
         ObjectSetInteger(0, text_name, OBJPROP_COLOR, marker_clr);
         ObjectSetInteger(0, text_name, OBJPROP_FONTSIZE, LabelFontSize);
         ObjectSetInteger(0, text_name, OBJPROP_ANCHOR, ANCHOR_LEFT_LOWER);
      }
      else
      {
         Print("DEBCO: ObjectCreate label failed. error=", GetLastError());
      }
   }

   ChartRedraw(0);
   Sleep(300);

   if(TakeScreenshots && StringLen(screenshot_name) > 0)
   {
      FolderCreate(ScreenshotFolder);

      string shot_path = ScreenshotFolder + "\\" + SafeFileName(screenshot_name);

      ResetLastError();

      bool ok = ChartScreenShot(0, shot_path, ScreenshotWidth, ScreenshotHeight, ALIGN_RIGHT);

      if(ok)
         Print("DEBCO screenshot saved: MQL5/Files/", shot_path);
      else
         Print("DEBCO screenshot FAILED: ", shot_path, " error=", GetLastError());
   }

   string done_path = relative_path + ".done";

   ResetLastError();

   bool moved = FileMove(relative_path, FILE_COMMON, done_path, FILE_COMMON | FILE_REWRITE);

   if(moved)
      Print("DEBCO event processed: ", relative_path, " setup=", setup_id, " side=", side, " chart=", _Symbol);
   else
      Print("DEBCO event processed but cannot move to .done: ", relative_path, " error=", GetLastError());

   ChartRedraw(0);
}
