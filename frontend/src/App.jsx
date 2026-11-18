import { useCallback, useEffect, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import {
  Activity, AlertTriangle, BarChart3, CheckCircle2, Database, Download, History,
  LayoutDashboard, Search, ShieldAlert, SlidersHorizontal, Upload,
} from 'lucide-react'
import './App.css'

const ML_API = 'http://127.0.0.1:8000'
const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'transactions', label: 'Transactions', icon: Search },
  { id: 'alerts', label: 'Fraud alerts', icon: ShieldAlert },
  { id: 'check', label: 'Check transaction', icon: SlidersHorizontal },
  { id: 'upload', label: 'Score a CSV dataset', icon: Upload },
  { id: 'upload-history', label: 'Upload history', icon: History },
  { id: 'uploaded-results', label: 'Uploaded results', icon: Database },
  { id: 'models', label: 'Model comparison', icon: BarChart3 },
]
const featureNames = Array.from({ length: 28 }, (_, index) => `V${index + 1}`)
const number = (value) => Number(value || 0).toLocaleString()
const percent = (value) => `${(Number(value || 0) * 100).toFixed(2)}%`

async function getJson(path) {
  const response = await fetch(`${ML_API}${path}`)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed (${response.status})`)
  }
  return response.json()
}

function usePageData(path) {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [refresh, setRefresh] = useState(0)
  const [loadedPath, setLoadedPath] = useState(null)
  const reload = useCallback(async () => {
    setLoading(true)
    setError('')
    setRefresh((value) => value + 1)
  }, [])
  useEffect(() => {
    let active = true
    getJson(path)
      .then((value) => {
        if (active) {
          setData(value)
          setError('')
          setLoadedPath(path)
        }
      })
      .catch((requestError) => {
        if (active) {
          setError(requestError.message)
          setLoadedPath(path)
        }
      })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [path, refresh])
  return {
    data: loadedPath === path ? data : null,
    error: loadedPath === path ? error : '',
    loading: loading || loadedPath !== path,
    reload,
  }
}

function ErrorMessage({ children }) {
  return children ? <div className="error-banner" role="alert">{children}</div> : null
}

function Loading({ children = 'Loading data…' }) {
  return <div className="loading">{children}</div>
}

function Dashboard({ source, onSourceChange, onNavigate }) {
  const historicalUpload = source.startsWith('upload:')
  const isUploadSource = source === 'uploaded' || historicalUpload
  const uploadId = historicalUpload ? source.slice('upload:'.length) : null
  const endpoint = historicalUpload
    ? `/api/uploads/${uploadId}/dashboard`
    : source === 'uploaded' ? '/api/uploads/current/dashboard' : '/api/dashboard'
  const { data, error, loading, reload } = usePageData(endpoint)
  if (loading) return <Loading>Loading dataset analytics…</Loading>
  if (error) return <>
    <div className="page-heading"><div><p className="eyebrow">BANKING FRAUD DETECTION</p><h1>Dashboard</h1></div><select className="source-select" value={source} onChange={(event) => onSourceChange(event.target.value)}><option value="baseline">Notebook baseline</option><option value="uploaded">Latest uploaded dataset</option>{historicalUpload && <option value={source}>Selected historical upload</option>}</select></div>
    <ErrorMessage>{error}</ErrorMessage>
    <button onClick={reload}>Retry</button>
  </>

  if (isUploadSource) {
    const stats = [
      { title: 'Rows scored', value: number(data.total_transactions), foot: data.source_filename, icon: Database },
      { title: 'Marked suspicious', value: number(data.suspicious_transactions), foot: `${percent(data.suspicious_rate)} of uploaded rows`, icon: AlertTriangle },
      { title: 'Marked genuine', value: number(data.genuine_transactions), foot: `Threshold ${Number(data.threshold).toFixed(4)}`, icon: CheckCircle2 },
      { title: 'Actual fraud labels', value: data.has_actual_labels ? number(data.actual_fraud_transactions) : 'Not provided', foot: data.has_actual_labels ? `${number(data.labeled_transactions)} labeled rows` : 'Predictions do not require labels', icon: ShieldAlert },
    ]
    const classData = [
      { name: 'Genuine prediction', count: data.genuine_transactions },
      { name: 'Suspicious prediction', count: data.suspicious_transactions },
    ]
    const evaluation = data.evaluation
    return <>
      <div className="page-heading">
        <div><p className="eyebrow">UPLOAD RESULTS</p><h1>{data.source_filename}</h1>
        <p>{historicalUpload ? 'Viewing a saved upload from history.' : 'Charts and counts update from the latest uploaded file after scoring completes.'}</p></div>
      <select className="source-select" value={source} onChange={(event) => onSourceChange(event.target.value)}><option value="baseline">Notebook baseline</option><option value="uploaded">Latest uploaded dataset</option>{historicalUpload && <option value={source}>{data.source_filename}</option>}</select>
      </div>
      <div className="stat-grid">
        {stats.map(({ title, value, foot, icon: StatIcon }) => (
          <article className="stat-card" key={title}>
            <div className="stat-top"><span>{title}</span><StatIcon size={18} /></div>
            <strong>{value}</strong><small>{foot}</small>
          </article>
        ))}
      </div>
      <div className="chart-grid">
        <article className="panel">
          <h2>Uploaded transactions by hour</h2>
          <p>{data.has_actual_labels ? 'Actual fraud labels are available for comparison.' : 'Actual fraud is unknown because this file has no Class labels.'}</p>
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.hourly}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="hour" tickFormatter={(hour) => `${hour}:00`} />
                <YAxis /><Tooltip /><Legend />
                <Line type="monotone" dataKey="transactions" name="Transactions" stroke="#2563eb" dot={false} />
                <Line type="monotone" dataKey="flagged" name="Predicted suspicious" stroke="#f79009" dot={false} />
                {data.has_actual_labels && <Line type="monotone" dataKey="actual_fraud" name="Actual fraud (provided labels)" stroke="#d92d20" dot={false} />}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>
        <article className="panel">
          <h2>Prediction breakdown</h2>
          <p>Every uploaded row is classified as Suspicious or Genuine.</p>
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={classData} dataKey="count" nameKey="name" innerRadius={62} outerRadius={94} paddingAngle={3} label>
                  <Cell fill="#12b76a" /><Cell fill="#d92d20" />
                </Pie>
                <Tooltip formatter={(value) => number(value)} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="confusion"><span>Average fraud probability</span><strong>{percent(data.average_fraud_probability)}</strong><span>Threshold</span><strong>{percent(data.threshold)}</strong></div>
        </article>
      </div>
      {evaluation && <div className="stat-grid metric-grid">
        {[
          ['Precision', evaluation.precision],
          ['Recall', evaluation.recall],
          ['F1 score', evaluation.f1_score],
          ['Accuracy', evaluation.accuracy],
        ].map(([label, value]) => <article className="stat-card compact" key={label}><span>{label} (labeled rows)</span><strong>{percent(value)}</strong></article>)}
      </div>}
      <button className="primary" onClick={() => onNavigate('uploaded-results')}>View uploaded transaction results</button>
    </>
  }

  const stats = [
    { title: 'Transactions analyzed', value: number(data.total_transactions), foot: `${number(data.test_transactions)} held-out test rows`, icon: Database },
    { title: 'Fraud cases in dataset', value: number(data.fraud_transactions), foot: `${percent(data.fraud_rate)} of all transactions`, icon: AlertTriangle },
    { title: 'Test-set fraud recall', value: percent(data.metrics.recall), foot: `${data.test_true_positives} frauds detected`, icon: ShieldAlert },
    { title: 'Test-set precision', value: percent(data.metrics.precision), foot: `${data.test_false_positives} false alarms`, icon: CheckCircle2 },
  ]
  const classData = [
    { name: 'Genuine', count: data.genuine_transactions },
    { name: 'Fraud', count: data.fraud_transactions },
  ]

  return (
    <>
      <div className="page-heading">
        <div><p className="eyebrow">BANKING FRAUD DETECTION</p><h1>Model and dataset overview</h1>
          <p>Real dataset statistics and saved Random Forest evaluation results.</p></div>
        <div className="dashboard-actions"><span className="model-badge"><Activity size={16} /> Random Forest · threshold {Number(data.threshold).toFixed(4)}</span><select className="source-select" value={source} onChange={(event) => onSourceChange(event.target.value)}><option value="baseline">Notebook baseline</option><option value="uploaded">Latest uploaded dataset</option></select></div>
      </div>
      <ErrorMessage>{error}</ErrorMessage>
      <div className="stat-grid">
        {stats.map(({ title, value, foot, icon: Icon }) => (
          <article className="stat-card" key={title}>
            <div className="stat-top"><span>{title}</span><Icon size={18} /></div>
            <strong>{value}</strong><small>{foot}</small>
          </article>
        ))}
      </div>
      <div className="chart-grid">
        <article className="panel">
          <h2>Transactions and fraud cases by hour</h2>
          <p>Derived from the notebook’s held-out test set; time is relative to the first dataset transaction.</p>
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.hourly}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="hour" tickFormatter={(hour) => `${hour}:00`} />
                <YAxis /><Tooltip /><Legend />
                <Line type="monotone" dataKey="transactions" name="Test transactions" stroke="#2563eb" dot={false} />
                <Line type="monotone" dataKey="fraud" name="Actual fraud" stroke="#d92d20" dot={false} />
                <Line type="monotone" dataKey="flagged" name="Flagged by model" stroke="#f79009" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>
        <article className="panel">
          <h2>Class imbalance</h2>
          <p>The model is evaluated on a dataset where fraud is rare.</p>
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={classData} dataKey="count" nameKey="name" innerRadius={62} outerRadius={94} paddingAngle={3} label>
                  <Cell fill="#12b76a" /><Cell fill="#d92d20" />
                </Pie>
                <Tooltip formatter={(value) => number(value)} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="confusion"><span>Test confusion matrix</span><strong>TN {number(data.metrics.confusion_matrix[0][0])}</strong><strong>FP {number(data.metrics.confusion_matrix[0][1])}</strong><strong>FN {number(data.metrics.confusion_matrix[1][0])}</strong><strong>TP {number(data.metrics.confusion_matrix[1][1])}</strong></div>
        </article>
      </div>
      <div className="stat-grid metric-grid">
        {[
          ['F1 score', data.metrics.f1_score],
          ['ROC AUC', data.metrics.roc_auc],
          ['PR AUC', data.metrics.pr_auc],
          ['Accuracy', data.metrics.accuracy],
        ].map(([label, value]) => <article className="stat-card compact" key={label}><span>{label}</span><strong>{percent(value)}</strong></article>)}
      </div>
    </>
  )
}

function RecordsPage({ alertMode = false, uploadedMode = false, uploadId = null }) {
  const [page, setPage] = useState(0)
  const [status, setStatus] = useState(alertMode ? 'Suspicious' : 'ALL')
  const [search, setSearch] = useState('')
  const [loaded, setLoaded] = useState(null)
  const [requestError, setRequestError] = useState(null)
  const path = uploadedMode
    ? `${uploadId ? `/api/uploads/${uploadId}` : '/api/uploads/current'}/transactions?page=${page}&size=25&status=${status}`
    : alertMode
      ? `/api/alerts?page=${page}&size=25`
      : `/api/transactions?page=${page}&size=25&status=${status}&search=${encodeURIComponent(search)}`
  const result = loaded?.path === path ? loaded.value : null
  const error = requestError?.path === path ? requestError.message : ''
  const loading = !result && !error

  useEffect(() => {
    let active = true
    getJson(path).then((value) => { if (active) setLoaded({ path, value }) })
      .catch((error) => { if (active) setRequestError({ path, message: error.message }) })
    return () => { active = false }
  }, [path])

  return (
    <section className="panel">
      <div className="section-title"><div><h2>{uploadedMode ? 'Uploaded transaction results' : alertMode ? 'Predicted suspicious transactions' : 'Held-out test transactions'}</h2><p>{uploadedMode ? 'Rows scored from the selected CSV. Actual labels are shown only if included in the file.' : alertMode ? 'Model-flagged rows from the saved test predictions; actual labels are included for evaluation.' : 'Saved notebook predictions with transaction features and model scores.'}</p></div></div>
      {!alertMode && !uploadedMode && <div className="filters">
        <input aria-label="Search time or amount" placeholder="Search time or amount" value={search} onChange={(event) => { setSearch(event.target.value); setPage(0) }} />
        <select value={status} onChange={(event) => { setStatus(event.target.value); setPage(0) }}><option>ALL</option><option>Genuine</option><option>Suspicious</option></select>
      </div>}
      {uploadedMode && <div className="filters"><select value={status} onChange={(event) => { setStatus(event.target.value); setPage(0) }}><option>ALL</option><option>Genuine</option><option>Suspicious</option></select></div>}
      <ErrorMessage>{error}</ErrorMessage>
      {loading ? <Loading /> : result && <>
        <div className="table-scroll"><table><thead><tr><th>Reference</th><th>Time</th><th>Amount</th><th>Probability</th><th>Prediction</th><th>Actual label</th></tr></thead>
          <tbody>{result.items.map((row) => <tr key={row.txn_ref}><td>{row.txn_ref}</td><td>{number(row.Time)}</td><td>${Number(row.Amount).toFixed(2)}</td><td>{percent(row.Fraud_Probability)}</td><td><span className={`pill ${row.Prediction_Status === 'Suspicious' ? 'danger' : 'safe'}`}>{row.Prediction_Status}</span></td><td>{row.Actual_Class == null ? 'Not provided' : Number(row.Actual_Class) === 1 ? 'Fraud' : 'Genuine'}</td></tr>)}</tbody>
        </table></div>
        <div className="pagination"><span>{number(result.total)} records</span><div><button disabled={page === 0} onClick={() => setPage((value) => value - 1)}>Previous</button><span>Page {page + 1} of {Math.max(result.pages, 1)}</span><button disabled={page + 1 >= result.pages} onClick={() => setPage((value) => value + 1)}>Next</button></div></div>
      </>}
    </section>
  )
}

function TransactionCheck() {
  const [values, setValues] = useState({ time: '0', amount: '0', ...Object.fromEntries(featureNames.map((name) => [name.toLowerCase(), '0'])) })
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [advanced, setAdvanced] = useState(true)

  const applySample = async (alert) => {
    setError('')
    try {
      const path = alert ? '/api/alerts?size=1' : '/api/transactions?status=Genuine&size=1'
      const data = await getJson(path)
      const row = data.items[0]
      if (!row) throw new Error('No matching sample transaction was found.')
      setValues({
        time: String(row.Time), amount: String(row.Amount),
        ...Object.fromEntries(featureNames.map((name) => [name.toLowerCase(), String(row[name] || 0)])),
      })
    } catch (requestError) { setError(requestError.message) }
  }

  const submit = async (event) => {
    event.preventDefault()
    setLoading(true); setError(''); setResult(null)
    const payload = Object.fromEntries(Object.entries(values).map(([key, value]) => [key, Number(value)]))
    try {
      const response = await fetch(`${ML_API}/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail || `Prediction failed (${response.status})`)
      setResult(body)
    } catch (requestError) { setError(requestError.message) }
    finally { setLoading(false) }
  }

  return <section className="panel">
    <h2>Score a transaction</h2><p>Enter Time, Amount and all 28 anonymized features. The saved Random Forest model returns the fraud probability.</p>
    <ErrorMessage>{error}</ErrorMessage>
    <div className="sample-actions"><button onClick={() => applySample(true)}>Load sample suspicious</button><button onClick={() => applySample(false)}>Load sample genuine</button><button onClick={() => setAdvanced((value) => !value)}>{advanced ? 'Hide' : 'Show'} V1–V28</button></div>
    <form onSubmit={submit}>
      <div className="feature-grid">
        <label>Time<input type="number" min="0" step="any" value={values.time} onChange={(event) => setValues({ ...values, time: event.target.value })} required /></label>
        <label>Amount<input type="number" min="0" step="any" value={values.amount} onChange={(event) => setValues({ ...values, amount: event.target.value })} required /></label>
        {advanced && featureNames.map((feature) => <label key={feature}>{feature}<input type="number" step="any" value={values[feature.toLowerCase()]} onChange={(event) => setValues({ ...values, [feature.toLowerCase()]: event.target.value })} required /></label>)}
      </div>
      <button className="primary" disabled={loading}>{loading ? 'Scoring…' : 'Run fraud prediction'}</button>
    </form>
    {result && <div className={`prediction-result ${result.prediction === 'Suspicious' ? 'result-danger' : 'result-safe'}`}><strong>{result.prediction}</strong><span>Probability: {percent(result.fraud_probability)}</span><span>Threshold: {percent(result.threshold)}</span></div>}
  </section>
}

function UploadDataset({ onUploaded }) {
  const [file, setFile] = useState(null)
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const scoreDataset = async (event) => {
    event.preventDefault()
    if (!file) return
    setLoading(true)
    setError('')
    setSummary(null)
    try {
      const response = await fetch(`${ML_API}/predict-csv?filename=${encodeURIComponent(file.name)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'text/csv' },
        body: file,
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || `Dataset scoring failed (${response.status})`)
      }

      const output = await response.blob()
      const downloadUrl = URL.createObjectURL(output)
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = 'scored_transactions.csv'
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000)

      setSummary({
        total: response.headers.get('X-Total-Transactions'),
        suspicious: response.headers.get('X-Suspicious-Transactions'),
        genuine: response.headers.get('X-Genuine-Transactions'),
      })
      onUploaded()
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoading(false)
    }
  }

  return <section className="panel upload-panel">
    <h2>Score a new transaction dataset</h2>
    <p>Upload a CSV and the saved Random Forest will score every row. The downloaded file includes fraud probability, prediction class, and prediction status.</p>
    <div className="schema-note">
      <strong>Required columns:</strong> Time, V1–V28, Amount. Column-name casing is ignored. Extra columns are kept in the output; an optional Class column is not used to make predictions.
      <br /><strong>Important:</strong> this model was trained on the ULB dataset’s anonymized V1–V28 features. A CSV containing only ordinary card fields (such as merchant, card number, or location) is not compatible.
    </div>
    <ErrorMessage>{error}</ErrorMessage>
    <form className="upload-form" onSubmit={scoreDataset}>
      <label htmlFor="dataset-csv">Choose a CSV file</label>
      <input id="dataset-csv" type="file" accept=".csv,text/csv" onChange={(event) => { setFile(event.target.files?.[0] || null); setSummary(null); setError('') }} />
      {file && <span className="file-selected">{file.name} · {(file.size / (1024 * 1024)).toFixed(2)} MB</span>}
      <button className="primary" type="submit" disabled={!file || loading}>{loading ? 'Scoring dataset…' : 'Score dataset and download results'}</button>
    </form>
    {summary && <div className="prediction-result result-safe">
      <strong>Dataset scored</strong>
      <span>{number(summary.total)} transactions</span>
      <span>{number(summary.suspicious)} marked Suspicious</span>
      <span>{number(summary.genuine)} marked Genuine</span>
      <span>Downloaded scored_transactions.csv</span>
    </div>}
  </section>
}

function ModelComparison() {
  const comparison = usePageData('/api/model-comparison')
  const importance = usePageData('/feature-importance')
  if (comparison.loading || importance.loading) return <Loading />
  if (comparison.error || importance.error) return <ErrorMessage>{comparison.error || importance.error}</ErrorMessage>
  const chartData = comparison.data.map((model) => ({
    model: model.Model,
    Precision: Number(model.Precision),
    Recall: Number(model.Recall),
    F1: Number(model['F1 Score']),
    'ROC-AUC': Number(model['ROC-AUC']),
    'PR-AUC': Number(model['PR-AUC']),
  }))
  return <div className="chart-grid">
    <article className="panel"><h2>Model comparison</h2><p>Notebook test-set results for the three evaluated classifiers.</p><div className="chart tall"><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="model" /><YAxis domain={[0, 1]} /><Tooltip /><Legend />{['Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC'].map((key, index) => <Bar key={key} dataKey={key} fill={['#2563eb', '#12b76a', '#7f56d9', '#06aed4', '#f79009'][index]} />)}</BarChart></ResponsiveContainer></div><div className="table-scroll"><table><thead><tr>{Object.keys(comparison.data[0] || {}).map((key) => <th key={key}>{key}</th>)}</tr></thead><tbody>{comparison.data.map((row) => <tr key={row.Model}>{Object.values(row).map((value, index) => <td key={index}>{typeof value === 'number' ? percent(value) : value}</td>)}</tr>)}</tbody></table></div></article>
    <article className="panel"><h2>Global feature importance</h2><p>Top features reported by the selected Random Forest.</p><div className="chart tall"><ResponsiveContainer width="100%" height="100%"><BarChart data={importance.data.slice(0, 15)} layout="vertical" margin={{ left: 24 }}><CartesianGrid strokeDasharray="3 3" horizontal={false} /><XAxis type="number" /><YAxis dataKey="Feature" type="category" width={50} /><Tooltip /><Bar dataKey="Importance" fill="#2563eb" /></BarChart></ResponsiveContainer></div></article>
  </div>
}

function UploadHistoryPage({ onViewUpload }) {
  const { data, error, loading } = usePageData('/api/uploads/history')
  if (loading) return <Loading>Loading upload history…</Loading>
  return <section className="panel">
    <div className="section-title">
      <div><h2>Scored CSV uploads</h2><p>Previously scored files remain available after refreshes and API restarts.</p></div>
    </div>
    <ErrorMessage>{error}</ErrorMessage>
    {!error && data.length === 0 && <div className="empty-history">No uploads yet. Score a compatible CSV to see it here.</div>}
    {!error && data.length > 0 && <div className="table-scroll">
      <table><thead><tr><th>File</th><th>Uploaded</th><th>Rows</th><th>Suspicious</th><th>Genuine</th><th>Actual fraud labels</th><th>Actions</th></tr></thead>
        <tbody>{data.map((upload) => <tr key={upload.upload_id}>
          <td>{upload.source_filename}</td>
          <td>{new Date(upload.uploaded_at).toLocaleString()}</td>
          <td>{number(upload.total_transactions)}</td>
          <td>{number(upload.suspicious_transactions)} ({percent(upload.suspicious_rate)})</td>
          <td>{number(upload.genuine_transactions)}</td>
          <td>{upload.has_actual_labels ? number(upload.actual_fraud_transactions) : 'Not provided'}</td>
          <td><div className="history-actions">
            <button type="button" onClick={() => onViewUpload(upload.upload_id)}>View</button>
            {upload.download_available
              ? <a href={`${ML_API}/api/uploads/${upload.upload_id}/download`}><Download size={14} /> Download</a>
              : <span className="unavailable">Scored file unavailable</span>}
          </div></td>
        </tr>)}</tbody>
      </table>
    </div>}
  </section>
}

export default function App() {
  const [active, setActive] = useState('dashboard')
  const [dashboardSource, setDashboardSource] = useState('baseline')
  const [selectedUploadId, setSelectedUploadId] = useState(null)
  const page = NAV_ITEMS.find((item) => item.id === active)
  const Icon = page.icon
  const navigateFromDashboard = (destination) => {
    if (destination === 'uploaded-results') setActive(destination)
    else {
      setDashboardSource(destination)
      setSelectedUploadId(destination.startsWith('upload:') ? destination.slice('upload:'.length) : null)
    }
  }
  const showLatestUpload = () => {
    setDashboardSource('uploaded')
    setSelectedUploadId(null)
    setActive('dashboard')
  }
  const viewHistoricalUpload = (uploadId) => {
    setDashboardSource(`upload:${uploadId}`)
    setSelectedUploadId(uploadId)
    setActive('dashboard')
  }
  return <div className="app-shell">
    <aside className="sidebar"><div className="brand"><span className="brand-mark">FD</span><span>Fraud Detection</span></div><p className="nav-caption">ANALYTICS</p><nav>{NAV_ITEMS.map(({ id, label, icon: ItemIcon }) => <button key={id} className={id === active ? 'nav-item active' : 'nav-item'} onClick={() => setActive(id)}><ItemIcon size={18} /><span>{label}</span></button>)}</nav><div className="sidebar-note">ULB credit card dataset<br />Random Forest model</div></aside>
    <main><header className="topbar"><div><span className="crumb">FRAUD ANALYTICS</span><h1><Icon size={21} />{page.label}</h1></div><a href={`${ML_API}/docs`} target="_blank" rel="noreferrer">ML API docs ↗</a></header>
      <div className="content">
        {active === 'dashboard' && <Dashboard source={dashboardSource} onSourceChange={navigateFromDashboard} onNavigate={setActive} />}
        {active === 'transactions' && <RecordsPage />}
        {active === 'alerts' && <RecordsPage alertMode />}
        {active === 'check' && <TransactionCheck />}
        {active === 'upload' && <UploadDataset onUploaded={showLatestUpload} />}
        {active === 'upload-history' && <UploadHistoryPage onViewUpload={viewHistoricalUpload} />}
        {active === 'uploaded-results' && <RecordsPage uploadedMode uploadId={selectedUploadId} />}
        {active === 'models' && <ModelComparison />}
        <footer>{dashboardSource === 'uploaded' || active === 'uploaded-results'
          ? 'Uploaded rows were scored by the saved model. Actual labels, when present in the CSV, are used only for evaluation.'
          : 'Baseline charts use the saved notebook model and ULB test-set output. Test labels are shown for evaluation.'}</footer>
      </div>
    </main>
  </div>
}
