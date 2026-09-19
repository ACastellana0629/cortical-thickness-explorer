import { useEffect, useState } from "react";

const STORAGE_KEY = "cortical-explorer-annotations";

function loadAnnotations() {
  try {
    const savedAnnotations = localStorage.getItem(STORAGE_KEY);
    return savedAnnotations ? JSON.parse(savedAnnotations) : [];
  } catch {
    return [];
  }
}

function AnnotationPanel({ selectedRegion, selectedGroup }) {
  const [annotations, setAnnotations] = useState(loadAnnotations);
  const [tag, setTag] = useState("Region of interest");
  const [note, setNote] = useState("");

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
  }, [annotations]);

  function saveAnnotation(event) {
    event.preventDefault();

    const trimmedNote = note.trim();

    const newAnnotation = {
      id: crypto.randomUUID(),
      regionId: selectedRegion.id,
      regionName: selectedRegion.displayName,
      group: selectedGroup,
      tag,
      note: trimmedNote,
      createdAt: new Date().toISOString(),
    };

    setAnnotations((currentAnnotations) => [
      newAnnotation,
      ...currentAnnotations,
    ]);

    setNote("");
  }

  function deleteAnnotation(annotationId) {
    setAnnotations((currentAnnotations) =>
      currentAnnotations.filter(
        (annotation) => annotation.id !== annotationId,
      ),
    );
  }

  return (
    <section className="annotation-section">
      <div className="annotation-heading">
        <div>
          <p className="section-label">Research annotation</p>
          <h3>Save region of interest</h3>
        </div>
        <span className="annotation-count">
          {annotations.length} saved
        </span>
      </div>

      <form className="annotation-form" onSubmit={saveAnnotation}>
        <label>
          Tag
          <select value={tag} onChange={(event) => setTag(event.target.value)}>
            <option>Region of interest</option>
            <option>Potential biomarker</option>
            <option>Review later</option>
            <option>Unexpected pattern</option>
            <option>Strong group difference</option>
          </select>
        </label>

        <label>
          Note
          <textarea
            value={note}
            onChange={(event) => setNote(event.target.value)}
            placeholder={`Add a note about ${selectedRegion.displayName}…`}
            rows="3"
          />
        </label>

        <button className="save-annotation-button" type="submit">
          Save annotation
        </button>
      </form>

      <div className="saved-annotations">
        <h3>Saved annotations</h3>

        {annotations.length === 0 ? (
          <p className="empty-annotations">
            Select a region and save a note to begin an ROI list.
          </p>
        ) : (
          <ul>
            {annotations.map((annotation) => (
              <li key={annotation.id}>
                <div>
                  <strong>{annotation.regionName}</strong>
                  <span>
                    {annotation.tag} ·{" "}
                    {annotation.group.replace("_", " ")}
                  </span>
                  {annotation.note && <p>{annotation.note}</p>}
                </div>

                <button
                  type="button"
                  onClick={() => deleteAnnotation(annotation.id)}
                  aria-label={`Delete annotation for ${annotation.regionName}`}
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}

export default AnnotationPanel;