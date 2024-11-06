package uob.oop;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Properties;


public class ArticlesEmbedding extends NewsArticles {
    private int intSize = -1;
    private String processedText = "";

    private INDArray newsEmbedding = Nd4j.create(0);

    public ArticlesEmbedding(String _title, String _content, NewsArticles.DataType _type, String _label) {
        //TODO Task 5.1 - 1 Mark
        super(_title,_content,_type,_label);
    }

    public void setEmbeddingSize(int _size) {
        intSize=_size;
        //TODO Task 5.2 - 0.5 Marks

    }

    public int getEmbeddingSize(){

        return intSize;
    }

    @Override
    public String getNewsContent() {
        if(processedText.isEmpty()){
            String clean = textCleaning(super.getNewsContent());
            String cleanLemmatized=lemmatization(clean);
            String finished = removeStopWords(cleanLemmatized,Toolkit.STOPWORDS);
            processedText= finished.toLowerCase();
            return processedText;
        }
        else {
            return processedText.trim();
        }
    }
    public static String lemmatization(String _clean){
        Properties prop = new Properties();
        prop.setProperty("annotators","tokenize,pos,lemma");
        StanfordCoreNLP pipelines = new StanfordCoreNLP(prop);
        CoreDocument documents = pipelines.processToCoreDocument(_clean);
        StringBuilder sb = new StringBuilder();
        for (CoreLabel t : documents.tokens()) {

            sb.append(t.lemma()).append(" ");
        }
        return sb.toString();
    }

    public static String removeStopWords(String _content, String[] _stopWords) {
        StringBuilder mySB = new StringBuilder();
        //TODO Task 2.3 - 3 marks
        String[] wordsList = _content.split(" ");
        for (String word : wordsList) {
            if (notContains(_stopWords, word)) {
                mySB.append(word).append(" ");
            }
        }

        return mySB.toString().trim();
    }

    private static boolean notContains(String[] _arrayTarget, String _searchValue) {
        for (String element : _arrayTarget) {
            if (_searchValue.equals(element)) {
                return false;
            }
        }
        return true;
    }

    public INDArray getEmbedding() throws Exception {
        //TODO Task 5.4 - 20 Marks
        // private INDArray newsEmbedding = Nd4j.create(0);
        if(newsEmbedding.isEmpty()){
            if(intSize==-1){
                throw new InvalidSizeException(" Invalid size ");
            }
            if (processedText.isEmpty()){
                throw new InvalidTextException(" Invalid Text ");
            }
            String[] processed = processedText.split(" ");
            newsEmbedding=Nd4j.zeros(intSize,50);
            int count = 0;
            for (int i = 0; i < processed.length ; i++) {
                Glove g = searchGlove(processed[i]);
                if(g!=null&&count<intSize){
                    INDArray vectorReP=Nd4j.create(g.getVector().getAllElements());
                    newsEmbedding.putRow(count,vectorReP);
                    count++;
                }
            }
            return Nd4j.vstack(newsEmbedding.mean(1));
        }
        else {
            return Nd4j.vstack(newsEmbedding.mean(1));
        }
    }
    private static Glove searchGlove(String _word){
         for (Glove g : AdvancedNewsClassifier.listGlove) {
             if ((g.getVocabulary().equalsIgnoreCase(_word))){
                 return g;
             }
         }
         return null;
     }
    /***
     * Clean the given (_content) text by removing all the characters that are not 'a'-'z', '0'-'9' and white space.
     * @param _content Text that need to be cleaned.
     * @return The cleaned text.
     */
    private static String textCleaning(String _content) {
        StringBuilder sbContent = new StringBuilder();

        for (char c : _content.toLowerCase().toCharArray()) {
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || Character.isWhitespace(c)) {
                sbContent.append(c);
            }
        }

        return sbContent.toString().trim();
    }
}
