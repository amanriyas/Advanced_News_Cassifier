package uob.oop;

import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AdvancedNewsClassifier {
    public Toolkit myTK = null;
    public static List<NewsArticles> listNews = null;
    public static List<Glove> listGlove = null;
    public List<ArticlesEmbedding> listEmbedding = null;
    public MultiLayerNetwork myNeuralNetwork = null;

    public final int BATCHSIZE = 10;

    public int embeddingSize = 0;
    private static StopWatch mySW = new StopWatch();

    public AdvancedNewsClassifier() throws IOException {
        myTK = new Toolkit();
        myTK.loadGlove();
        listNews = myTK.loadNews();
        listGlove = createGloveList();
        listEmbedding = loadData();
    }

    public static void main(String[] args) throws Exception {
        mySW.start();
        AdvancedNewsClassifier myANC = new AdvancedNewsClassifier();

        myANC.embeddingSize = myANC.calculateEmbeddingSize(myANC.listEmbedding);
        myANC.populateEmbedding();
        myANC.myNeuralNetwork = myANC.buildNeuralNetwork(2);
        myANC.predictResult(myANC.listEmbedding);
        myANC.printResults();
        mySW.stop();
        System.out.println("Total elapsed time: " + mySW.getTime());
    }

    public List<Glove> createGloveList() {
        List<Glove> listResult = new ArrayList<>();
        String [] stopWords = Toolkit.STOPWORDS;
        List<String> vocabulary = Toolkit.getListVocabulary();
        List<double[]> vectors = Toolkit.getlistVectors();
        int size = vocabulary.size();
        for (int i = 0; i <size ; i++) {
            String word = vocabulary.get(i);
            boolean IsStopWord = false;
            for (String stopword: stopWords) {
                if(word.equals(stopword)){
                    IsStopWord=true;
                    break;
                }
            }
            if(!IsStopWord){
                double[] vectorRepresenation = vectors.get(i);
                Vector vector = new Vector(vectorRepresenation);
                Glove glove = new Glove(word,vector);
                listResult.add(glove);
            }
        }
        //TODO Task 6.1 - 5 Marks

        return listResult;
    }


    public static List<ArticlesEmbedding> loadData() {
        List<ArticlesEmbedding> listEmbedding = new ArrayList<>();
        for (NewsArticles news : listNews) {
            ArticlesEmbedding myAE = new ArticlesEmbedding(news.getNewsTitle(), news.getNewsContent(), news.getNewsType(), news.getNewsLabel());
            listEmbedding.add(myAE);
        }
        return listEmbedding;
    }

    public int calculateEmbeddingSize(List<ArticlesEmbedding> _listEmbedding) {
        int intMedian = -1;
        List<Integer> lenghtsofDoc = new ArrayList<>();
        for (ArticlesEmbedding embedding :_listEmbedding) {
            String contents = embedding.getNewsContent();
            String[] content = contents.split("\\s+");
            int count=0;
            for (String word : content) {
                if(Toolkit.getListVocabulary().contains(word)&&FindGloveWord(word)!=null){
                    count++;
                }
            }
            lenghtsofDoc.add(count);
        }
        int size=lenghtsofDoc.size();
        int a = 1;
        while (a<size){
            int temp=lenghtsofDoc.get(a);
            int b = a-1;
            while (b>=0&&lenghtsofDoc.get(b)>temp){
                lenghtsofDoc.set(b+1,lenghtsofDoc.get(b));
                --b;
            }
            lenghtsofDoc.set(b+1,temp);
            ++a;
        }
        int middle = size/2;
        if(size%2==0){
            intMedian=(lenghtsofDoc.get(middle)+lenghtsofDoc.get(middle+1))/2;
        }
        else{
            intMedian=lenghtsofDoc.get(middle);
        }

        //TODO Task 6.2 - 5 Marks

        return intMedian;
    }
    public static Glove FindGloveWord(String _word){
        for (Glove g : AdvancedNewsClassifier.listGlove) {
            if(g.getVocabulary().equalsIgnoreCase(_word)){
                return g;
            }
        }
        return null;
    }
    public void populateEmbedding() {
        //TODO Task 6.3 - 10 Marks
        for (ArticlesEmbedding embedding :listEmbedding) {
            try{
                embedding.getEmbedding();
            }catch (InvalidSizeException e){
                embedding.setEmbeddingSize(embeddingSize);
            }catch (InvalidTextException e){
                embedding.getNewsContent();
            }catch (Exception e){
                System.out.println(e.getMessage());
            }
        }
    }

    public DataSetIterator populateRecordReaders(int _numberOfClasses) throws Exception {
        ListDataSetIterator myDataIterator = null; //TODO Task 6.4 - 8 Marks
        List<DataSet> listDS = new ArrayList<>();
        INDArray inputNDArray = null;
        INDArray outputNDArray = null;
        try{
            for (ArticlesEmbedding embedding : listEmbedding) {
                String type = embedding.getNewsType().toString();
                if(type.equals("Training")){
                    inputNDArray= embedding.getEmbedding();
                    outputNDArray=Nd4j.zeros(1,_numberOfClasses);
                    int labelno = Integer.parseInt(embedding.getNewsLabel());
                    for (int i = 1; i <=_numberOfClasses ; i++) {
                        if(labelno==i){
                            outputNDArray.putScalar(i-1,1);
                        }
                    }
                    DataSet myDataset = new DataSet(inputNDArray,outputNDArray);
                    listDS.add(myDataset);
                }
            }
        }catch (Exception e){
            System.out.println(e.getMessage());
        }
        return new ListDataSetIterator(listDS, BATCHSIZE);
    }

    public MultiLayerNetwork buildNeuralNetwork(int _numOfClasses) throws Exception {
        DataSetIterator trainIter = populateRecordReaders(_numOfClasses);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Adam.builder().learningRate(0.02).beta1(0.9).beta2(0.999).build())
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(embeddingSize).nOut(15)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.HINGE)
                        .activation(Activation.SOFTMAX)
                        .nIn(15).nOut(_numOfClasses).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        for (int n = 0; n < 100; n++) {
            model.fit(trainIter);
            trainIter.reset();
        }
        return model;
    }

    public List<Integer> predictResult(List<ArticlesEmbedding> _listEmbedding) throws Exception {
        List<Integer> listResult = new ArrayList<>();
        //TODO Task 6.5 - 8 Marks
        try{
            for (ArticlesEmbedding embedding: _listEmbedding) {
                if(!(embedding.getNewsType().toString().equals("Training"))){
                    INDArray docEmbedding = embedding.getEmbedding();
                    int[] labels = myNeuralNetwork.predict(docEmbedding);
                    listResult.add(labels[0]);
                    embedding.setNewsLabel(String.valueOf(labels[0]));
                }
            }
        }catch (Exception e){
            System.out.println(e.getMessage());
        }
        return listResult;
    }
    public void printResults() {
        //TODO Task 6.6 - 6.5 Marks
        List<List<String>> grouping = new ArrayList<>();
        for (ArticlesEmbedding embedding: listEmbedding){
            if(embedding.getNewsType().toString().equals("Testing")){
                int labels = Integer.parseInt(embedding.getNewsLabel());
                while (grouping.size()<=labels){
                    grouping.add(new ArrayList<>());
                }
                grouping.get(labels).add(embedding.getNewsTitle());
            }
        }
        for (int i = 0; i <grouping.size() ; i++) {
            System.out.println("Group " + (i+1));
            for (String Title : grouping.get(i)){
                System.out.println(Title);
            }
        }
    }
}
