import React from 'react';
import styles from './styles.module.css';

import CodeBlock from '@theme/CodeBlock';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

const bibtexCitation = `@inproceedings{skyline-yu20,
  title = {{Skyline: Interactive In-Editor Computational Performance Profiling
    for Deep Neural Network Training}},
  author = {Yu, Geoffrey X. and Grossman, Tovi and Pekhimenko, Gennady},
  booktitle = {{Proceedings of the 33rd ACM Symposium on User Interface
    Software and Technology (UIST'20)}},
  year = {2020},
}`;

const textCitation = `Geoffrey X. Yu, Tovi Grossman, and Gennady Pekhimenko.
Skyline: Interactive In-Editor Computational Performance Profiling for
Deep Neural Network Training. In Proceedings of the 33rd ACM Symposium
on User Interface Software and Technology (UIST'20). 2020.`;

function CitationBlock() {
  return (
    <div className={styles.citationBlock}>
      <Tabs
        defaultValue="bibtex"
        values={[
          { label: 'BibTeX', value: 'bibtex', },
          { label: 'Plain Text', value: 'txt', },
        ]}
      >
        <TabItem value="bibtex">
          <CodeBlock language="tex">{bibtexCitation}</CodeBlock>
        </TabItem>
        <TabItem value="txt">
          <CodeBlock language="txt">{textCitation}</CodeBlock>
        </TabItem>
      </Tabs>
    </div>
  );
}

export default CitationBlock;
