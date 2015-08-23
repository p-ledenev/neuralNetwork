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
public class NetworkTester {

    BasicNetwork network;
    MLDataSet dataSet;

    @Before
    public void setUp() throws Throwable {

        dataSet = new CSVNeuralDataSet(Runner.dataPath + "usd_50inputs_max-min_2010.txt", 50, 2, false, new CSVFormat('.', ';'), false);
        network = (BasicNetwork) new PerceptronBuilder().read();
    }

    @Test
    public void printTotalNetworkError() throws Throwable {

        dataSet = new CSVNeuralDataSet(Runner.dataPath + "usd_sample.txt", 50, 2, false, new CSVFormat('.', ';'), false);
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
            Log.info((i++) + " Tutor value: " + print(pair.getIdeal()) + "; Network value: " + print(output) +
                    "; totalNetworkError: " + network.calculateError(set));
        }
    }

    private String print(MLData data) {

        String result = "";
        for (double d : data.getData())
            result += Round.toSignificant(d) + ";";

        return result;
    }
}
