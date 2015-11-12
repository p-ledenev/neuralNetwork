import model.Log;
import model.Round;
import model.Runner;
import networks.PerceptronBuilder;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.data.specific.CSVNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.util.csv.CSVFormat;
import org.junit.Before;
import org.junit.Test;

/**
 * Created by DiKey on 22.08.2015.
 */
public class NetworkTrainigTests {

    BasicNetwork network;
    MLDataSet dataSet;

    @Before
    public void setUp() throws Throwable {

        String fileName;
        //fileName = "usd_50inputs_max-min_2010.txt";
        fileName = "usd_app.txt";

        dataSet = new CSVNeuralDataSet(Runner.dataPath + fileName, 100, 3, false, new CSVFormat('.', ';'), false);
        network = (BasicNetwork) new PerceptronBuilder().read();
    }

    @Test
    public void printTotalNetworkError() throws Throwable {

        printResults();

        Log.info("Total error on tutor set: " + network.calculateError(dataSet));
    }

    @Test
    public void printResults() throws Throwable {

        int i = 0;
        for (MLDataPair pair : dataSet) {
            MLDataSet set = new BasicMLDataSet();
            set.add(pair);

            MLData output = network.compute(pair.getInput());

            if (isBuySignal(pair.getIdeal()) || isSellSignal(pair.getIdeal()))
                Log.info((i++) + " Tutor value: " + print(pair.getIdeal()) + "; Network value: " + print(output) +
                        "; totalNetworkError: " + network.calculateError(set));
        }
    }

    private boolean isBuySignal(MLData output) {
        return output.getData(0) == 1;
    }

    private boolean isSellSignal(MLData output) {
        return output.getData(1) == 1;
    }

    private String print(MLData data) {

        String result = "";
        for (double d : data.getData())
            result += Round.toSignificant(d) + ";";

        return result;
    }
}
