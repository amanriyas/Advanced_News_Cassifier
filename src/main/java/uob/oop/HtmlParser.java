package uob.oop;

public class HtmlParser {
    /***
     * Extract the title of the news from the _htmlCode.
     * @param _htmlCode Contains the full HTML string from a specific news. E.g. 01.htm.
     * @return Return the title if it's been found. Otherwise, return "Title not found!".
     */
    public static String getNewsTitle(String _htmlCode) {
        String titleTagOpen = "<title>";
        String titleTagClose = "</title>";

        int titleStart = _htmlCode.indexOf(titleTagOpen) + titleTagOpen.length();
        int titleEnd = _htmlCode.indexOf(titleTagClose);

        if (titleStart != -1 && titleEnd != -1 && titleEnd > titleStart) {
            String strFullTitle = _htmlCode.substring(titleStart, titleEnd);
            return strFullTitle.substring(0, strFullTitle.indexOf(" |"));
        }

        return "Title not found!";
    }

    /***
     * Extract the content of the news from the _htmlCode.
     * @param _htmlCode Contains the full HTML string from a specific news. E.g. 01.htm.
     * @return Return the content if it's been found. Otherwise, return "Content not found!".
     */
    public static String getNewsContent(String _htmlCode) {
        String contentTagOpen = "\"articleBody\": \"";
        String contentTagClose = " \",\"mainEntityOfPage\":";

        int contentStart = _htmlCode.indexOf(contentTagOpen) + contentTagOpen.length();
        int contentEnd = _htmlCode.indexOf(contentTagClose);

        if (contentStart != -1 && contentEnd != -1 && contentEnd > contentStart) {
            return _htmlCode.substring(contentStart, contentEnd).toLowerCase();
        }

        return "Content not found!";
    }

    public static NewsArticles.DataType getDataType(String _htmlCode) {
        String openTag = "<datatype>";
        String closeTag = "</datatype>";
        int contentStart = _htmlCode.indexOf(openTag) + openTag.length();
        int contentClose = _htmlCode.indexOf(closeTag) + closeTag.length();
        if (contentStart != -1 && contentClose != -1 && contentClose > contentStart) {
            if (_htmlCode.substring(contentStart, contentClose).contains("Training")) {
                return NewsArticles.DataType.Training;
            }

        }
        return NewsArticles.DataType.Testing;
        //TODO Task 3.1 - 1.5 Marks
    }

    public static String getLabel (String _htmlCode) {
        String firstTag="<label>";
        String lastTag="</label>";
        if(_htmlCode.contains(firstTag)&&_htmlCode.contains(lastTag)) {
            return _htmlCode.substring(_htmlCode.indexOf(firstTag)+firstTag.length(),_htmlCode.indexOf(lastTag));//TODO Task 3.2 - 1.5 Marks
        }
        else {
            return "-1";
        }

         //Please modify the return value.
    }


}
