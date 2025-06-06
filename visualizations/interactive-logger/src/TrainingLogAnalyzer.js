import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Upload, FileText, BarChart3, Download } from "lucide-react";

const TrainingLogAnalyzer = () => {
  const [data, setData] = useState([]);
  const [fileName, setFileName] = useState("log.txt");
  const [error, setError] = useState("");
  const [activeChart, setActiveChart] = useState("accuracy");

  // Initial data from the provided log
  const initialLogData = `{"train_lr": 9.9999999999991e-07, "train_loss": 6.922781705689563, "test_loss": 6.9021368203339755, "test_acc1": 0.20600001268386842, "test_acc5": 0.936000042629242, "epoch": 0, "n_parameters": 7148008}
{"train_lr": 9.9999999999991e-07, "train_loss": 6.906692454187895, "test_loss": 6.878793769412571, "test_acc1": 0.36000001487731936, "test_acc5": 1.3880000759124755, "epoch": 1, "n_parameters": 7148008}
{"train_lr": 2.5799999999998435e-05, "train_loss": 6.830090122781307, "test_loss": 6.527459109271014, "test_acc1": 1.3500000806808472, "test_acc5": 4.956000265502929, "epoch": 2, "n_parameters": 7148008}
{"train_lr": 5.060000000000849e-05, "train_loss": 6.697745880396437, "test_loss": 6.130513332508229, "test_acc1": 3.168000171661377, "test_acc5": 9.91400048828125, "epoch": 3, "n_parameters": 7148008}
{"train_lr": 7.539999999998438e-05, "train_loss": 6.554651913668612, "test_loss": 5.7658969561258955, "test_acc1": 5.230000225830078, "test_acc5": 15.31000078125, "epoch": 4, "n_parameters": 7148008}
{"train_lr": 0.00010019999999999544, "train_loss": 6.455093737414701, "test_loss": 5.491449726952447, "test_acc1": 7.440000384521484, "test_acc5": 20.054001123046874, "epoch": 5, "n_parameters": 7148008}
{"train_lr": 0.00012492119824836773, "train_loss": 6.372584407754558, "test_loss": 5.320285231978805, "test_acc1": 8.564000436401367, "test_acc5": 22.474001123046875, "epoch": 6, "n_parameters": 7148008}
{"train_lr": 0.0001248865368846039, "train_loss": 6.287473566216721, "test_loss": 5.049385724244295, "test_acc1": 11.400000720214845, "test_acc5": 27.926001538085938, "epoch": 7, "n_parameters": 7148008}
{"train_lr": 0.00012484558244039054, "train_loss": 6.202164600435778, "test_loss": 4.836443141654685, "test_acc1": 13.684000610351562, "test_acc5": 32.06000166015625, "epoch": 8, "n_parameters": 7148008}
{"train_lr": 0.0001247983394068342, "train_loss": 6.121352361141444, "test_loss": 4.581240795276783, "test_acc1": 15.998000793457031, "test_acc5": 35.62400192871094, "epoch": 9, "n_parameters": 7148008}
{"train_lr": 0.00012474481296465557, "train_loss": 6.045008978659777, "test_loss": 4.4530016404611095, "test_acc1": 17.85000079345703, "test_acc5": 38.64800222167969, "epoch": 10, "n_parameters": 7148008}
{"train_lr": 0.00012468500898368665, "train_loss": 5.973522756597121, "test_loss": 4.24926550300033, "test_acc1": 20.164001147460937, "test_acc5": 42.14800246582031, "epoch": 11, "n_parameters": 7148008}
{"train_lr": 0.00012461893402204168, "train_loss": 5.900049332615664, "test_loss": 4.125982840855916, "test_acc1": 21.556000915527342, "test_acc5": 44.23600244140625, "epoch": 12, "n_parameters": 7148008}
{"train_lr": 0.00012454659532559309, "train_loss": 5.849310526840216, "test_loss": 3.980146187323111, "test_acc1": 23.384001000976564, "test_acc5": 46.99000288085937, "epoch": 13, "n_parameters": 7148008}
{"train_lr": 0.00012446800082712683, "train_loss": 5.780270549366705, "test_loss": 3.8548392807995833, "test_acc1": 25.256001342773438, "test_acc5": 49.38400263671875, "epoch": 14, "n_parameters": 7148008}
{"train_lr": 0.0001243831591453807, "train_loss": 5.738724355479415, "test_loss": 3.7376536704875805, "test_acc1": 26.7800009765625, "test_acc5": 51.3540029296875, "epoch": 15, "n_parameters": 7148008}
{"train_lr": 0.00012429207958422544, "train_loss": 5.693068793947272, "test_loss": 3.6999226499486855, "test_acc1": 27.380001440429687, "test_acc5": 52.2140025390625, "epoch": 16, "n_parameters": 7148008}
{"train_lr": 0.00012419477213154763, "train_loss": 5.6447090688798065, "test_loss": 3.5588383498015226, "test_acc1": 29.190001416015626, "test_acc5": 54.51200263671875, "epoch": 17, "n_parameters": 7148008}
{"train_lr": 0.00012409124745825102, "train_loss": 5.609748727459606, "test_loss": 3.5102183553907604, "test_acc1": 29.980001318359374, "test_acc5": 55.2620025390625, "epoch": 18, "n_parameters": 7148008}
{"train_lr": 0.00012398151691692688, "train_loss": 5.565813474541755, "test_loss": 3.413316594229804, "test_acc1": 31.51800168457031, "test_acc5": 57.1580025390625, "epoch": 19, "n_parameters": 7148008}
{"train_lr": 0.00012386559254076206, "train_loss": 5.535867127678473, "test_loss": 3.3429532845815024, "test_acc1": 32.52000126953125, "test_acc5": 58.47200307617187, "epoch": 20, "n_parameters": 7148008}
{"train_lr": 0.00012374348704222694, "train_loss": 5.5076525471955655, "test_loss": 3.320107680779916, "test_acc1": 32.922001782226566, "test_acc5": 59.11400302734375, "epoch": 21, "n_parameters": 7148008}
{"train_lr": 0.00012361521381147673, "train_loss": 5.47431985595577, "test_loss": 3.2757423277254456, "test_acc1": 33.500002099609375, "test_acc5": 59.55000327148438, "epoch": 22, "n_parameters": 7148008}
{"train_lr": 0.00012348078691519688, "train_loss": 5.455401881707372, "test_loss": 3.209032244152493, "test_acc1": 34.87600178222656, "test_acc5": 60.9280025390625, "epoch": 23, "n_parameters": 7148008}
{"train_lr": 0.00012334022109479823, "train_loss": 5.422357753657704, "test_loss": 3.159410927030775, "test_acc1": 35.620002001953125, "test_acc5": 61.742003125, "epoch": 24, "n_parameters": 7148008}
{"train_lr": 0.00012319353176491336, "train_loss": 5.397703431981454, "test_loss": 3.1055762502882214, "test_acc1": 36.33200187988281, "test_acc5": 62.5600029296875, "epoch": 25, "n_parameters": 7148008}
{"train_lr": 0.0001230407350116134, "train_loss": 5.373721923449819, "test_loss": 3.086202515496148, "test_acc1": 37.00400183105469, "test_acc5": 63.19800307617187, "epoch": 26, "n_parameters": 7148008}
{"train_lr": 0.00012288184759088716, "train_loss": 5.356642449943187, "test_loss": 3.0456363006874367, "test_acc1": 37.714001879882815, "test_acc5": 63.88400258789063, "epoch": 27, "n_parameters": 7148008}
{"train_lr": 0.0001227168869264393, "train_loss": 5.337768382853646, "test_loss": 2.9951976316946523, "test_acc1": 37.856002026367186, "test_acc5": 64.06800263671875, "epoch": 28, "n_parameters": 7148008}
{"train_lr": 0.00012254587110806998, "train_loss": 5.327127774604124, "test_loss": 2.975894777863114, "test_acc1": 38.57400239257812, "test_acc5": 64.640002734375, "epoch": 29, "n_parameters": 7148008}
{"train_lr": 0.00012236881888968, "train_loss": 5.302780654159286, "test_loss": 2.9459581816637956, "test_acc1": 38.8520021484375, "test_acc5": 65.26600317382812, "epoch": 30, "n_parameters": 7148008}
{"train_lr": 0.00012218574968694295, "train_loss": 5.285112329905362, "test_loss": 2.914485498710915, "test_acc1": 39.40600270996094, "test_acc5": 65.82200244140626, "epoch": 31, "n_parameters": 7148008}
{"train_lr": 0.00012199668357555639, "train_loss": 5.26619555888702, "test_loss": 2.869403349028693, "test_acc1": 39.88200256347656, "test_acc5": 65.90000307617187, "epoch": 32, "n_parameters": 7148008}
{"train_lr": 0.00012180164128865164, "train_loss": 5.251431771915117, "test_loss": 2.840738393642284, "test_acc1": 40.6100025390625, "test_acc5": 66.482002734375, "epoch": 33, "n_parameters": 7148008}
{"train_lr": 0.00012160064421485225, "train_loss": 5.238648565171911, "test_loss": 2.826367484198676, "test_acc1": 40.75800258789062, "test_acc5": 67.21000288085938, "epoch": 34, "n_parameters": 7148008}
{"train_lr": 0.00012139371439578959, "train_loss": 5.227222876321021, "test_loss": 2.803957294534754, "test_acc1": 41.41600283203125, "test_acc5": 67.84000341796875, "epoch": 35, "n_parameters": 7148008}
{"train_lr": 0.00012118087452360146, "train_loss": 5.203176718059299, "test_loss": 2.782798917205245, "test_acc1": 41.20400234375, "test_acc5": 67.81800322265624, "epoch": 36, "n_parameters": 7148008}
{"train_lr": 0.00012096214793855793, "train_loss": 5.195209150632127, "test_loss": 2.781951974939417, "test_acc1": 41.36000278320312, "test_acc5": 67.70400341796875, "epoch": 37, "n_parameters": 7148008}
{"train_lr": 0.0001207375862658480, "train_loss": 5.184730062882106, "test_loss": 2.7569607231352062, "test_acc1": 42.15200180664063, "test_acc5": 68.25800258789063, "epoch": 38, "n_parameters": 7148008}
{"train_lr": 0.00012050713121633858, "train_loss": 5.170153301241967, "test_loss": 2.735135661231147, "test_acc1": 42.158002490234374, "test_acc5": 68.52600283203125, "epoch": 39, "n_parameters": 7148008}
{"train_lr": 0.00012027089097685828, "train_loss": 5.158535007438023, "test_loss": 2.7200133049929582, "test_acc1": 42.658002490234374, "test_acc5": 69.07000327148438, "epoch": 40, "n_parameters": 7148008}
{"train_lr": 0.00012002886381443891, "train_loss": 5.144969793222696, "test_loss": 2.6962195060871266, "test_acc1": 43.0100021484375, "test_acc5": 69.32200283203125, "epoch": 41, "n_parameters": 7148008}
{"train_lr": 0.00011978107627021738, "train_loss": 5.137973815595789, "test_loss": 2.66055009983204, "test_acc1": 43.4980021484375, "test_acc5": 69.8760033203125, "epoch": 42, "n_parameters": 7148008}
{"train_lr": 0.00011952755551680601, "train_loss": 5.12988906918908, "test_loss": 2.6625191591404103, "test_acc1": 43.61800244140625, "test_acc5": 69.95000405273437, "epoch": 43, "n_parameters": 7148008}
{"train_lr": 0.00011926832935560953, "train_loss": 5.118051862068695, "test_loss": 2.5987005322067827, "test_acc1": 44.73600244140625, "test_acc5": 70.61400322265625, "epoch": 44, "n_parameters": 7148008}
{"train_lr": 0.0001190034262137549, "train_loss": 5.109512474480198, "test_loss": 2.6192186541027493, "test_acc1": 44.48400288085937, "test_acc5": 70.478002734375, "epoch": 45, "n_parameters": 7148008}
{"train_lr": 0.00011873287514084143, "train_loss": 5.095237688730947, "test_loss": 2.6224158030969127, "test_acc1": 44.164002783203124, "test_acc5": 70.37800341796876, "epoch": 46, "n_parameters": 7148008}
{"train_lr": 0.00011845670580580919, "train_loss": 5.089821927970071, "test_loss": 2.589891791343689, "test_acc1": 44.68800288085937, "test_acc5": 70.87000346679687, "epoch": 47, "n_parameters": 7148008}
{"train_lr": 0.00011817494849374975, "train_loss": 5.068483112872743, "test_loss": 2.5651880635155573, "test_acc1": 45.24800268554687, "test_acc5": 71.314002734375, "epoch": 48, "n_parameters": 7148008}
{"train_lr": 0.00011788763410251521, "train_loss": 5.065270316615093, "test_loss": 2.593982555248119, "test_acc1": 44.79400268554687, "test_acc5": 71.07200302734375, "epoch": 49, "n_parameters": 7148008}
{"train_lr": 0.00011759479413941841, "train_loss": 5.053361165687907, "test_loss": 2.550087496086403, "test_acc1": 45.42800283203125, "test_acc5": 71.53600244140625, "epoch": 50, "n_parameters": 7148008}
{"train_lr": 0.00011729646071759982, "train_loss": 5.04944785032198, "test_loss": 2.5396696152510465, "test_acc1": 45.4580021484375, "test_acc5": 71.632003125, "epoch": 51, "n_parameters": 7148008}
{"train_lr": 0.00011699266655273839, "train_loss": 5.046868064730383, "test_loss": 2.5287676917182074, "test_acc1": 45.83800263671875, "test_acc5": 72.1600029296875, "epoch": 52, "n_parameters": 7148008}
{"train_lr": 0.00011668344495922515, "train_loss": 5.035204007483119, "test_loss": 2.539989568569042, "test_acc1": 45.93000229492188, "test_acc5": 71.78400268554688, "epoch": 53, "n_parameters": 7148008}
{"train_lr": 0.00011636882984674553, "train_loss": 5.028561175929652, "test_loss": 2.4864912739506475, "test_acc1": 46.4520025390625, "test_acc5": 72.56200390625, "epoch": 54, "n_parameters": 7148008}
{"train_lr": 0.00011604885571636633, "train_loss": 5.021355886420281, "test_loss": 2.507577547320613, "test_acc1": 46.314002392578125, "test_acc5": 72.4400033203125, "epoch": 55, "n_parameters": 7148008}
{"train_lr": 0.00011572355765685746, "train_loss": 5.015070599653452, "test_loss": 2.4955169845510414, "test_acc1": 46.51600234375, "test_acc5": 72.4100033203125, "epoch": 56, "n_parameters": 7148008}
{"train_lr": 0.00011539297134081638, "train_loss": 5.007223039126987, "test_loss": 2.532897366417779, "test_acc1": 46.23600239257812, "test_acc5": 72.10800366210937, "epoch": 57, "n_parameters": 7148008}
{"train_lr": 0.00011505713302077367, "train_loss": 4.998119787799178, "test_loss": 2.467882129881117, "test_acc1": 46.78400219726562, "test_acc5": 72.886004296875, "epoch": 58, "n_parameters": 7148008}
{"train_lr": 0.00011471607952518015, "train_loss": 4.987156302880326, "test_loss": 2.4778388562025846, "test_acc1": 46.65400283203125, "test_acc5": 72.864003515625, "epoch": 59, "n_parameters": 7148008}
{"train_lr": 0.00011436984825440849, "train_loss": 4.9809610364939285, "test_loss": 2.443031827608744, "test_acc1": 47.250002099609375, "test_acc5": 73.07800390625, "epoch": 60, "n_parameters": 7148008}
{"train_lr": 0.00011401847717655084, "train_loss": 4.984159991526298, "test_loss": 2.434249167089109, "test_acc1": 47.43400209960937, "test_acc5": 73.162003515625, "epoch": 61, "n_parameters": 7148008}
{"train_lr": 0.00011366200482345821, "train_loss": 4.974230938213621, "test_loss": 2.463952422142029, "test_acc1": 46.98800249023437, "test_acc5": 72.88000380859376, "epoch": 62, "n_parameters": 7148008}
{"train_lr": 0.00011330047028638434, "train_loss": 4.968018625315716, "test_loss": 2.460732866216589, "test_acc1": 46.992002880859374, "test_acc5": 73.32000380859375, "epoch": 63, "n_parameters": 7148008}
{"train_lr": 0.00011293391321161743, "train_loss": 4.963217988514977, "test_loss": 2.42778601469817, "test_acc1": 47.482002392578124, "test_acc5": 73.5920033203125, "epoch": 64, "n_parameters": 7148008}
{"train_lr": 0.00011256237379623111, "train_loss": 4.953011351845247, "test_loss": 2.4169502037542836, "test_acc1": 47.798002197265625, "test_acc5": 73.69000419921875, "epoch": 65, "n_parameters": 7148008}
{"train_lr": 0.00011218589278374736, "train_loss": 4.948826903681294, "test_loss": 2.4126471678415933, "test_acc1": 47.748002880859374, "test_acc5": 73.9200044921875, "epoch": 66, "n_parameters": 7148008}
{"train_lr": 0.00011180451145962568, "train_loss": 4.938520552502643, "test_loss": 2.4323607462423817, "test_acc1": 48.166002587890624, "test_acc5": 73.9400044921875, "epoch": 67, "n_parameters": 7148008}
{"train_lr": 0.00011141827164649558, "train_loss": 4.946345944961102, "test_loss": 2.38952127650932, "test_acc1": 48.714002587890626, "test_acc5": 74.65600390625, "epoch": 68, "n_parameters": 7148008}
{"train_lr": 0.00011102721569998215, "train_loss": 4.926305802546436, "test_loss": 2.3775071921171964, "test_acc1": 48.80400239257813, "test_acc5": 74.468004296875, "epoch": 69, "n_parameters": 7148008}
{"train_lr": 0.00011063138650377642, "train_loss": 4.923084887085582, "test_loss": 2.387129033053363, "test_acc1": 48.82000219726562, "test_acc5": 74.33600458984375, "epoch": 70, "n_parameters": 7148008}
{"train_lr": 0.00011023082746496838, "train_loss": 4.924470004203508, "test_loss": 2.3606101583551475, "test_acc1": 48.826002685546875, "test_acc5": 74.486004296875, "epoch": 71, "n_parameters": 7148008}
{"train_lr": 0.00010982558250938201, "train_loss": 4.918575577384276, "test_loss": 2.3349404025960854, "test_acc1": 49.16600234375, "test_acc5": 74.91200458984375, "epoch": 72, "n_parameters": 7148008}
{"train_lr": 0.00010941569607672689, "train_loss": 4.907666455534437, "test_loss": 2.3396827114952936, "test_acc1": 49.2300025390625, "test_acc5": 74.74800380859375, "epoch": 73, "n_parameters": 7148008}
{"train_lr": 0.00010900121311564896, "train_loss": 4.903301924681492, "test_loss": 2.361502753363715, "test_acc1": 49.31000258789062, "test_acc5": 74.84800390625, "epoch": 74, "n_parameters": 7148008}
{"train_lr": 0.00010858217907886544, "train_loss": 4.908938155507298, "test_loss": 2.362476044230991, "test_acc1": 48.9140025390625, "test_acc5": 74.49200419921875, "epoch": 75, "n_parameters": 7148008}`;

  // Parse log data function
  const parseLogData = (logText) => {
    try {
      const lines = logText.trim().split("\n");
      const parsed = lines
        .map((line) => {
          try {
            return JSON.parse(line);
          } catch (e) {
            return null;
          }
        })
        .filter((item) => item !== null);

      return parsed.map((item) => ({
        epoch: item.epoch,
        trainLoss: parseFloat(item.train_loss.toFixed(4)),
        testLoss: parseFloat(item.test_loss.toFixed(4)),
        testAcc1: parseFloat(item.test_acc1.toFixed(2)),
        testAcc5: parseFloat(item.test_acc5.toFixed(2)),
        trainLr: parseFloat(item.train_lr.toExponential(3)),
        nParameters: item.n_parameters,
      }));
    } catch (error) {
      setError("Error parsing log data: " + error.message);
      return [];
    }
  };

  // Load initial data
  useEffect(() => {
    const parsedData = parseLogData(initialLogData);
    setData(parsedData);
  }, []);

  // Handle file upload
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      setError("");
      setFileName(file.name);

      const text = await file.text();
      const parsedData = parseLogData(text);

      if (parsedData.length === 0) {
        setError("No valid JSON lines found in the file");
        return;
      }

      setData(parsedData);
    } catch (error) {
      setError("Error reading file: " + error.message);
    }
  };

  // Export data as CSV
  const exportToCSV = () => {
    if (data.length === 0) return;

    const headers = [
      "Epoch",
      "Train Loss",
      "Test Loss",
      "Test Acc@1",
      "Test Acc@5",
      "Learning Rate",
      "Parameters",
    ];
    const csvContent = [
      headers.join(","),
      ...data.map((row) =>
        [
          row.epoch,
          row.trainLoss,
          row.testLoss,
          row.testAcc1,
          row.testAcc5,
          row.trainLr,
          row.nParameters,
        ].join(",")
      ),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "training_log.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const renderChart = () => {
    if (data.length === 0) return null;

    switch (activeChart) {
      case "accuracy":
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip formatter={(value, name) => [value + "%", name]} />
              <Legend />
              <Line
                type="monotone"
                dataKey="testAcc1"
                stroke="#8884d8"
                strokeWidth={2}
                name="Top-1 Accuracy"
                dot={{ r: 3 }}
              />
              <Line
                type="monotone"
                dataKey="testAcc5"
                stroke="#82ca9d"
                strokeWidth={2}
                name="Top-5 Accuracy"
                dot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        );

      case "loss":
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="trainLoss"
                stroke="#ff7300"
                strokeWidth={2}
                name="Training Loss"
                dot={{ r: 3 }}
              />
              <Line
                type="monotone"
                dataKey="testLoss"
                stroke="#ff0000"
                strokeWidth={2}
                name="Test Loss"
                dot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        );

      case "lr":
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis tickFormatter={(value) => value.toExponential(1)} />
              <Tooltip
                formatter={(value) => [value.toExponential(3), "Learning Rate"]}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="trainLr"
                stroke="#8884d8"
                strokeWidth={2}
                name="Learning Rate"
                dot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-800 mb-2">
                Training Log Analyzer
              </h1>
              <p className="text-gray-600">
                Visualize and analyze deep learning training metrics
              </p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={exportToCSV}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                disabled={data.length === 0}
              >
                <Download size={16} />
                Export CSV
              </button>
            </div>
          </div>
        </div>

        {/* File Upload */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center gap-4 mb-4">
            <FileText className="text-blue-600" size={24} />
            <h2 className="text-xl font-semibold text-gray-800">
              Import Training Log
            </h2>
          </div>

          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors cursor-pointer">
              <Upload size={16} />
              Choose File
              <input
                type="file"
                accept=".txt,.log,.json"
                onChange={handleFileUpload}
                className="hidden"
              />
            </label>
            <span className="text-gray-600">Current file: {fileName}</span>
          </div>

          {error && (
            <div className="mt-4 p-3 bg-red-100 border border-red-300 rounded-lg text-red-700">
              {error}
            </div>
          )}

          <div className="mt-4 text-sm text-gray-500">
            <p>
              Upload a log file with JSON lines containing training metrics.
            </p>
            <p>
              Expected format: {"{"}"epoch": 0, "train_loss": 1.23, "test_loss":
              1.45, "test_acc1": 67.8, "test_acc5": 89.2, ...{"}"}
            </p>
          </div>
        </div>

        {/* Chart Controls */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center gap-4 mb-4">
            <BarChart3 className="text-purple-600" size={24} />
            <h2 className="text-xl font-semibold text-gray-800">
              Visualization
            </h2>
          </div>

          <div className="flex gap-3">
            <button
              onClick={() => setActiveChart("accuracy")}
              className={`px-4 py-2 rounded-lg transition-colors ${
                activeChart === "accuracy"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-200 text-gray-700 hover:bg-gray-300"
              }`}
            >
              Accuracy
            </button>
            <button
              onClick={() => setActiveChart("loss")}
              className={`px-4 py-2 rounded-lg transition-colors ${
                activeChart === "loss"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-200 text-gray-700 hover:bg-gray-300"
              }`}
            >
              Loss
            </button>
            <button
              onClick={() => setActiveChart("lr")}
              className={`px-4 py-2 rounded-lg transition-colors ${
                activeChart === "lr"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-200 text-gray-700 hover:bg-gray-300"
              }`}
            >
              Learning Rate
            </button>
          </div>
        </div>

        {/* Chart */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          {data.length > 0 ? (
            <>
              <h3 className="text-lg font-semibold text-gray-800 mb-4">
                {activeChart === "accuracy" && "Test Accuracy Over Epochs"}
                {activeChart === "loss" && "Training & Test Loss Over Epochs"}
                {activeChart === "lr" && "Learning Rate Schedule"}
              </h3>
              {renderChart()}
            </>
          ) : (
            <div className="text-center py-12 text-gray-500">
              <BarChart3 size={48} className="mx-auto mb-4 opacity-50" />
              <p>No data to display. Please upload a training log file.</p>
            </div>
          )}
        </div>

        {/* Data Table */}
        {data.length > 0 && (
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              Training Data Summary
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-2 px-3">Epoch</th>
                    <th className="text-left py-2 px-3">Train Loss</th>
                    <th className="text-left py-2 px-3">Test Loss</th>
                    <th className="text-left py-2 px-3">Top-1 Acc (%)</th>
                    <th className="text-left py-2 px-3">Top-5 Acc (%)</th>
                    <th className="text-left py-2 px-3">Learning Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {data.slice(0, 10).map((row, idx) => (
                    <tr
                      key={idx}
                      className="border-b border-gray-100 hover:bg-gray-50"
                    >
                      <td className="py-2 px-3">{row.epoch}</td>
                      <td className="py-2 px-3">{row.trainLoss}</td>
                      <td className="py-2 px-3">{row.testLoss}</td>
                      <td className="py-2 px-3">{row.testAcc1}</td>
                      <td className="py-2 px-3">{row.testAcc5}</td>
                      <td className="py-2 px-3">
                        {row.trainLr.toExponential(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {data.length > 10 && (
                <p className="text-gray-500 text-center py-3">
                  Showing first 10 rows of {data.length} total epochs
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TrainingLogAnalyzer;
