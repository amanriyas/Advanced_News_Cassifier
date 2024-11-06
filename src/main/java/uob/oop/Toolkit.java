package uob.oop;

import java.io.*;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

public class Toolkit {
    public static List<String> listVocabulary= null;
    public static List<double[]> listVectors= null;
    private static final String FILENAME_GLOVE = "glove.6B.50d_Reduced.csv";

    public static final String[] STOPWORDS = {"a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your"};

    public void loadGlove() throws IOException {
        BufferedReader myReader ;
        try {
             myReader=new BufferedReader(new FileReader(getFileFromResource(FILENAME_GLOVE)));
             listVocabulary=new ArrayList<>();
             listVectors= new ArrayList<>();
             String s = null;
             while((s= myReader.readLine())!=null){
                     String[] string=s.split(",");
                     listVocabulary.add(string[0]);
                     double[] vectorRepresentation= new double[string.length-1];
                     for (int i = 1; i < string.length ; i++) {
                         vectorRepresentation[i-1]=Double.parseDouble(string[i]);
                     }
                     listVectors.add(vectorRepresentation);
                 } //TODO Task 4.1 - 5 marks
          } catch (URISyntaxException e) {
            System.out.println(e.getMessage());
        }

    }

    private static File getFileFromResource(String fileName) throws URISyntaxException {
        ClassLoader classLoader = Toolkit.class.getClassLoader();
        URL resource = classLoader.getResource(fileName);
        if (resource == null) {
            throw new IllegalArgumentException(fileName);
        } else {
            return new File(resource.toURI());
        }
    }

    public List<NewsArticles> loadNews() {
        List<NewsArticles> listNews = new ArrayList<>();
        try (Stream<Path> paths = Files.walk(Paths.get("src/main/resources"))) {
            paths.filter(Files::isRegularFile).filter(p -> p.toString().endsWith(".htm"))
                    .sorted(Comparator.comparing(Path::getFileName))
                    .forEach(p -> {
                   try{
                       String Html = Files.readString(p);
                       String title= HtmlParser.getNewsTitle(Html);
                       String content = HtmlParser.getNewsContent(Html);
                       NewsArticles.DataType type = HtmlParser.getDataType(Html);
                       String label = HtmlParser.getLabel(Html);
                       NewsArticles article = new NewsArticles(title,content,type,label);
                       listNews.add(article);
                   }catch (IOException e){
                       e.getMessage();
                   }

                    });
        } catch (IOException e) {
            e.getMessage();
        }
        //TODO Task 4.2 - 5 Marks
        return listNews;
    }

    public static List<String> getListVocabulary() {
        return listVocabulary;
    }

    public static List<double[]> getlistVectors() {
        return listVectors;
    }
}
